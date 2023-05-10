import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from .utils import plot_tensor, denormalize_residual, load_model
from .data import create_dataset
from . import config 


def logging(logger, original_spec, synthesized_spec, target_residual, noisy_spec, pred, mask, title, step):
    zero_indexes = (mask == 0).nonzero()
    if len(zero_indexes):
        start_zero_index = int(zero_indexes[0][-1])
    else:
        start_zero_index = original_spec.shape[-1]
    original_spec = original_spec[:,:start_zero_index]
    synthesized_spec = synthesized_spec[:,:start_zero_index]
    noisy_spec = noisy_spec[:,:start_zero_index]
    if config.model_type1 == "spec2residual":
        target_residual = target_residual[:,:start_zero_index]
        pred_residual = pred[:,:start_zero_index]
        if config.normallize_residual:
            pred_spec = denormalize_residual(pred[:,:start_zero_index]) + synthesized_spec
        else:
            pred_spec = pred[:,:start_zero_index] + synthesized_spec
        logger.add_image(f'{title}/target_residual_spec', plot_tensor(target_residual.squeeze().cpu().detach().numpy(), "residual"), global_step=step, dataformats='HWC')
        logger.add_image(f'{title}/predicted_residual_spec', plot_tensor(pred_residual.squeeze().cpu().detach().numpy(), "residual"), global_step=step, dataformats='HWC')
    else:
        pred_spec = pred[:,:start_zero_index]
    
    logger.add_image(f'{title}/input_spec', plot_tensor(synthesized_spec.squeeze().cpu().detach().numpy(), "spectrum"), global_step=step, dataformats='HWC')
    logger.add_image(f'{title}/predicted_spec', plot_tensor(pred_spec.squeeze().cpu().detach().numpy(), "spectrum"), global_step=step, dataformats='HWC')
    logger.add_image(f'{title}/target_spec', plot_tensor(original_spec.squeeze().cpu().detach().numpy(), "spectrum"), global_step=step, dataformats='HWC')
    logger.add_image(f'{title}/noisy_spec', plot_tensor(noisy_spec.squeeze().cpu().detach().numpy(), "noisy_spectrum"), global_step=step, dataformats='HWC')


def resgrad_train(args):
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_model_path, exist_ok=True)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=config.log_dir)
    print("Load data...")
    train_dataset, val_dataset = create_dataset()
    
    print("Load model...")
    model, optimizer = load_model(train=True, restore_model_epoch=args.restore_epoch)
    
    scaler = torch.cuda.amp.GradScaler()

    step = 0
    avg_val_loss = 0
    avg_train_loss = 0
    print("Start training...")
    for epoch in range(config.epochs):
        loop = tqdm(train_dataset)
        train_loss_list = []
        for train_data in loop:
            step += 1
            if config.model_type1 == "spec2residual":
                synthesized_spec, original_spec, residual_spec, mask = train_data
                synthesized_spec = synthesized_spec.to(config.device)
                mask = mask.to(config.device)
                residual_spec = residual_spec.to(config.device)
                loss, pred = model.compute_loss(residual_spec, mask, synthesized_spec)
            else:
                synthesized_spec, original_spec, mask = train_data
                mask = mask.to(config.device)
                synthesized_spec = synthesized_spec.to(config.device)
                original_spec = original_spec.to(config.device)
                loss, pred = model.compute_loss(original_spec, mask, synthesized_spec)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_list.append(loss.item())

            if step % config.validate_every_n_step == 0:
                model.eval()
                with torch.no_grad():
                    all_val_loss = []
                    val_num = 0
                    for val_data in val_dataset:
                        val_num += 1
                        if config.model_type1 == "spec2residual":
                            synthesized_spec, original_spec, target_residual, mask = val_data
                            synthesized_spec = synthesized_spec.to(config.device)
                            mask = mask.to(config.device)
                            target_residual = target_residual.to(config.device)
                            val_loss, noisy_spec = model.compute_loss(target_residual, mask, synthesized_spec)
                        else:
                            synthesized_spec, original_spec, mask = val_data
                            synthesized_spec = synthesized_spec.to(config.device)
                            mask = mask.to(config.device)
                            original_spec = original_spec.to(config.device)
                            val_loss, noisy_spec = model.compute_loss(original_spec, mask, synthesized_spec)
                            target_residual = [None for _ in range(len(original_spec))]
                            
                        all_val_loss.append(val_loss.item())
                    
                        ## logging result spectrums
                        if val_num == 1:
                            z = synthesized_spec + torch.randn_like(synthesized_spec, device=config.device) / 1.5
                            # Generate sample by performing reverse dynamics
                            pred = model(z, mask, synthesized_spec, n_timesteps=50, stoc=False, spk=None)
                            for i in range(3):
                                logging(logger, original_spec[i], synthesized_spec[i], target_residual[i], noisy_spec[i], pred[i], mask[i], \
                                        f'image{i}_epoch{epoch}', step)

                avg_val_loss = sum(all_val_loss) / len(all_val_loss)
                logger.add_scalar('validation/loss', avg_val_loss,  global_step=step)
                avg_train_loss = sum(train_loss_list) / len(train_loss_list)
                logger.add_scalar('training/loss', avg_train_loss,  global_step=step)
                train_loss_list = []
                model.train()

            loop.set_description(f'Epoch {epoch}, Step {step}: ')
            loop.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)
        
        ## Save checkpoints
        torch.save(model.state_dict(), os.path.join(config.save_path, f'ResGrad_epoch{epoch}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(config.save_path, 'optimizer.pth'))
   

