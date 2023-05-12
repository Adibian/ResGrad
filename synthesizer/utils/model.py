import os
import torch

from ..model import FastSpeech2, ScheduledOptim


def get_model(restore_step, config, train=False):

    device = config['synthesizer']['main']['device']
    model = FastSpeech2(config).to(device)
    if restore_step:
        ckpt_path = os.path.join(
            config['synthesizer']['train']["path"]["ckpt_path"],
            "{}.pth.tar".format(restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=torch.device(device))
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, config['synthesizer']['train'], config['synthesizer']['model'], restore_step
        )
        if restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

