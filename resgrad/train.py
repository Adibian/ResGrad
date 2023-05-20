import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from .utils import plot_tensor, denormalize_residual, load_model
from .data import create_dataset


def logging(logger, config, original_spec, synthesized_spec, target_residual, noisy_spec, pred, mask, title, step):
    zero_indexes = (mask == 0).nonzero()
    if len(zero_indexes):
        start_zero_index = int(zero_indexes[0][-1])
    else:
        start_zero_index = original_spec.shape[-1]
    original_spec = original_spec[:,:start_zero_index]
    synthesized_spec = synthesized_spec[:,:start_zero_index]
    noisy_spec = noisy_spec[:,:start_zero_index]
    if config['model']['model_type1'] == "spec2residual":
        target_residual = target_residual[:,:start_zero_index]
        pred_residual = pred[:,:start_zero_index]
        if config['data']['normallize_residual']:
            pred_spec = denormalize_residual(pred[:,:start_zero_index], config) + synthesized_spec
        else:
            pred_spec = pred[:,:start_zero_index] + synthesized_spec
        logger.add_image(f'{title}/target_residual_spec', plot_tensor(target_residual.squeeze().cpu().detach().numpy(), "residual", config), global_step=step, dataformats='HWC')
        logger.add_image(f'{title}/predicted_residual_spec', plot_tensor(pred_residual.squeeze().cpu().detach().numpy(), "residual", config), global_step=step, dataformats='HWC')
    else:
        pred_spec = pred[:,:start_zero_index]
    
    logger.add_image(f'{title}/input_spec', plot_tensor(synthesized_spec.squeeze().cpu().detach().numpy(), "spectrum", config), global_step=step, dataformats='HWC')
    logger.add_image(f'{title}/predicted_spec', plot_tensor(pred_spec.squeeze().cpu().detach().numpy(), "spectrum", config), global_step=step, dataformats='HWC')
    logger.add_image(f'{title}/target_spec', plot_tensor(original_spec.squeeze().cpu().detach().numpy(), "spectrum", config), global_step=step, dataformats='HWC')
    logger.add_image(f'{title}/noisy_spec', plot_tensor(noisy_spec.squeeze().cpu().detach().numpy(), "noisy_spectrum", config), global_step=step, dataformats='HWC')


def resgrad_train(args, config):
    os.makedirs(config['train']['log_dir'], exist_ok=True)
    os.makedirs(config['train']['save_model_path'], exist_ok=True)

    device = config['main']['device']

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=config['train']['log_dir'])
    print("Load data...")
    train_dataset, val_dataset = create_dataset(config)
    
    print("Load model...")
    model, optimizer = load_model(config, train=True, restore_model_step=args.restore_step)
    
    scaler = torch.cuda.amp.GradScaler()
    # grad_acc_step = config["optimizer"]["grad_acc_step"]
    # grad_clip_thresh = config["optimizer"]["grad_clip_thresh"]

    step = args.restore_step - 1
    epoch = args.restore_step // (len(train_dataset)//config['data']['batch_size'] + 1)
    avg_val_loss = 0
    avg_train_loss = 0

    print("Start training...")
    outer_bar = tqdm(total=config['train']['total_steps'], desc="Total Training", position=0)
    outer_bar.n = step  
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(train_dataset), desc="Epoch {}".format(epoch), position=1)
        train_loss_list = []
        epoch += 1
        for train_data in train_dataset:
            step += 1
            inner_bar.update(1)
            outer_bar.update(1)
            if config['model']['model_type1'] == "spec2residual":
                synthesized_spec, original_spec, residual_spec, mask, speakers = train_data
                synthesized_spec = synthesized_spec.to(device)
                mask = mask.to(device)
                residual_spec = residual_spec.to(device)
                if config['main']['multi_speaker']:
                    speakers = speakers.to(device)
                    loss, pred = model.compute_loss(residual_spec, mask, synthesized_spec, speakers)
                else:
                    loss, pred = model.compute_loss(residual_spec, mask, synthesized_spec)

            else:
                synthesized_spec, original_spec, mask, speakers = train_data
                mask = mask.to(device)
                synthesized_spec = synthesized_spec.to(device)
                original_spec = original_spec.to(device)
                if config['main']['multi_speaker']:
                    speakers = speakers.to(device)
                    loss, pred = model.compute_loss(original_spec, mask, synthesized_spec, speakers)
                else:
                    loss, pred = model.compute_loss(original_spec, mask, synthesized_spec)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_list.append(loss.item())

            # loss.backward()
            # train_loss_list.append(loss.item())
            # if step % grad_acc_step == 0:
            #     # Clipping gradients to avoid gradient explosion
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
            #     # Update weights
            #     optimizer.step_and_update_lr()
            #     optimizer.zero_grad()

            if step % config['train']['validate_step'] == 0:
                model.eval()
                with torch.no_grad():
                    all_val_loss = []
                    val_num = 0
                    for val_data in val_dataset:
                        val_num += 1
                        if config['model']['model_type1'] == "spec2residual":
                            synthesized_spec, original_spec, target_residual, mask, speakers = val_data
                            synthesized_spec = synthesized_spec.to(device)
                            mask = mask.to(device)
                            target_residual = target_residual.to(device)
                            if config['main']['multi_speaker']:
                                speakers = speakers.to(device)
                                val_loss, noisy_spec = model.compute_loss(target_residual, mask, synthesized_spec, speakers)
                            else:
                                val_loss, noisy_spec = model.compute_loss(target_residual, mask, synthesized_spec)
                                
                        else:
                            synthesized_spec, original_spec, mask, speakers = val_data
                            synthesized_spec = synthesized_spec.to(device)
                            mask = mask.to(device)
                            original_spec = original_spec.to(device)
                            if config['main']['multi_speaker']:
                                speakers = speakers.to(device)
                                val_loss, noisy_spec = model.compute_loss(original_spec, mask, synthesized_spec, speakers)
                            else:
                                val_loss, noisy_spec = model.compute_loss(original_spec, mask, synthesized_spec)
                            target_residual = [None for _ in range(len(original_spec))]
                        all_val_loss.append(val_loss.item())
                    
                        ## logging result spectrums
                        if val_num == 1:
                            z = synthesized_spec + torch.randn_like(synthesized_spec, device=device) / 1.5
                            # Generate sample by performing reverse dynamics
                            if config['main']['multi_speaker']:
                                pred = model(z, mask, synthesized_spec, n_timesteps=50, stoc=False, spk=speakers)
                            else:
                                pred = model(z, mask, synthesized_spec, n_timesteps=50, stoc=False, spk=None)
                            for i in range(3):
                                logging(logger, config, original_spec[i], synthesized_spec[i], target_residual[i], noisy_spec[i], pred[i], mask[i], \
                                        f'image{i}_step{step}', step)

                avg_val_loss = sum(all_val_loss) / len(all_val_loss)
                logger.add_scalar('validation/loss', avg_val_loss,  global_step=step)
                avg_train_loss = sum(train_loss_list) / len(train_loss_list)
                logger.add_scalar('training/loss', avg_train_loss,  global_step=step)
                train_loss_list = []
                model.train()

            inner_bar.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)

            if step % config['train']['save_ckpt_step'] == 0:
                ## Save checkpoints
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                        # "optimizer": optimizer._optimizer.state_dict(),
                    },
                    os.path.join(config['train']['save_model_path'], f'ResGrad_step{step}.pth')
                )
                # torch.save(model.state_dict(), os.path.join(config['train']['save_model_path'], f'ResGrad_step{step}.pth'))
                # torch.save(optimizer.state_dict(), os.path.join(config['train']['save_model_path'], 'optimizer.pth'))

            if step > config['train']['total_steps']:
                quit()

   

