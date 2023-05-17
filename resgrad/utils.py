import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from .model import Diffusion
from .model.optimizer import ScheduledOptim

def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_tensor(tensor, tensor_name, config):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 3))
    if tensor_name == "spectrum" and config['data']['normallize_spectrum']:
        im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none', \
                        vmin = 0.2, vmax = 0.9)
    elif tensor_name == "residual" and config['data']['normallize_residual']:
        im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none', \
                        vmin = 0.2, vmax = 0.9)
    else:
        im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def plot_spectrum(spec, path):
    plt.figure(figsize=(10, 3))
    im = plt.imshow(spec, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def crop_masked_values(mat_list, length):
    new_mat_list = []
    for mat in mat_list:
        new_mat_list.append(mat[:,:length])
    return new_mat_list

def normalize_data(data, config):
    if config['data']['normalized_method'] == "min-max":
        data = (data - config['data']['min_spec_value'])/(config['data']['max_spec_value'] - config['data']['min_spec_value'])
    else:
        print("normalized method is not supported!")
    return data

def normalize_residual(residual_spec, config):
    if config['data']['normalized_method'] == "min-max":
        residual_spec = (residual_spec - config['data']['min_residual_value'])/(config['data']['max_residual_value'] - \
                                                                                    config['data']['min_residual_value'])
    else:
        print("normalized method is not supported!")
    return residual_spec

def denormalize_data(data, config):
    if config['data']['normalized_method'] == "min-max":
        data = data * (config['data']['max_spec_value'] - config['data']['min_spec_value']) + config['data']['min_spec_value']
    else:
        print("normalized method is not supported!")
    return data

def denormalize_residual(residual_spec, config):
    if config['data']['normalized_method'] == "min-max":
        residual_spec = residual_spec * (config['data']['max_residual_value'] - config['data']['min_residual_value']) + \
                                                                                        config['data']['min_residual_value']
    else:
        print("normalized method is not supported!")
    return residual_spec

def load_model(config, train=False, restore_model_step=0):
    model = Diffusion(n_feats=config['model']['n_feats'], dim=config['model']['dim'], n_spks=config['model']['n_spks'], \
                      spk_emb_dim=config['model']['spk_emb_dim'], beta_min=config['model']['beta_min'], \
                      beta_max=config['model']['beta_max'], pe_scale=config['model']['pe_scale'])
    model = model.to(config['main']['device'])
    if restore_model_step > 0:
        checkpoint = torch.load(os.path.join(config['train']['save_model_path'], f'ResGrad_step{restore_model_step}.pth'), \
                                map_location=config['main']['device'])
        # checkpoint = torch.load(os.path.join("/mnt/hdd1/adibian/FastSpeech2/ResGrad/output/Persian/dur_taget_pitch_target/resgrad/ckpt", \
        #                                      f'ResGrad_epoch{restore_model_epoch}.pth'), \
        #                         map_location=config['main']['device'])
        # model.load_state_dict(checkpoint["model"])
        model.load_state_dict(checkpoint['model'])

    if train:
        # restore_step = 670*restore_model_step ## 670 is number of steps per epoch
        scheduled_optim = ScheduledOptim(model, config, restore_model_step)
        # optimizer = torch.optim.Adam(params=model.parameters(), lr=config['train']['lr'])
        if restore_model_step > 0:
            # optimizer_state = torch.load(os.path.join(config['train']['save_model_path'], 'optimizer.pth'))
            scheduled_optim.load_state_dict(checkpoint['optimizer'])
            # optimizer.load_state_dict(optimizer_state)
        model.train()
        return model, scheduled_optim

    model.eval()        
    return model