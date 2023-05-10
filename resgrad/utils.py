import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from . import config
from .model import Diffusion


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_tensor(tensor, tensor_name):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 3))
    if tensor_name == "spectrum" and config.normallize_spectrum:
        im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none', \
                        vmin = 0.2, vmax = 0.9)
    elif tensor_name == "residual" and config.normallize_residual:
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

def normalize_data(data):
    if config.normalized_method == "min-max":
        data = (data - config.min_spec_value)/(config.max_spec_value - config.min_spec_value)
    else:
        print("normalized method is not supported!")
    return data

def normalize_residual(residual_spec):
    if config.normalized_method == "min-max":
        residual_spec = (residual_spec - config.min_residual_value)/(config.max_residual_value - config.min_residual_value)
    else:
        print("normalized method is not supported!")
    return residual_spec

def denormalize_data(data):
    if config.normalized_method == "min-max":
        data = data * (config.max_spec_value - config.min_spec_value) + config.min_spec_value
    else:
        print("normalized method is not supported!")
    return data

def denormalize_residual(residual_spec):
    if config.normalized_method == "min-max":
        residual_spec = residual_spec * (config.max_residual_value - config.min_residual_value) + config.min_residual_value
    else:
        print("normalized method is not supported!")
    return residual_spec

def load_model(train=False, restore_model_epoch=0):
    model = Diffusion(n_feats=config.n_feats, dim=config.dim, n_spks=config.n_spks, spk_emb_dim=config.spk_emb_dim, 
                        beta_min=config.beta_min, beta_max=config.beta_max, pe_scale=config.pe_scale)
    model = model.to(config.device)

    if train:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    
    if restore_model_epoch > 0:
        checkpoint = torch.load(os.join(config.save_path, f'ResGrad{restore_model_epoch}.pth'), map_location=config.device)
        model.load_state_dict(checkpoint)
        if train:
            optimizer_state = torch.load(os.join(config.save_path, 'optimizer.pth'))
            optimizer.load_state_dict(optimizer_state)

    if train:
        return model, optimizer
    else:
        model.eval()        
        return model