
from vocoder import params
from vocoder.models import Generator
import json
import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def get_vocoder(restore_step=0):
    with open("vocoder/ckpt/config.json", "r") as f:
        config = json.load(f)
    config = AttrDict(config)
    vocoder = Generator(config)

    if restore_step:
        ckpt = torch.load(f"vocoder/ckpt/g_{restore_step}", map_location=params.device)
    else:
        ckpt = torch.load(f"vocoder/ckpt/{params.model_name}", map_location=params.device)

    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(params.device)
    return vocoder

