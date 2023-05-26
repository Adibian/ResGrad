
from vocoder.models import Generator
import json
import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def get_vocoder(config, device):
    with open("vocoder/ckpt/config.json", "r") as f:
        model_config = json.load(f)
    model_config = AttrDict(model_config)
    vocoder = Generator(model_config)

    # if config['restore_step']:
    #     ckpt = torch.load(f"vocoder/ckpt/g_{config['restore_step']}", map_location=config['device'])
    # else:
    ckpt = torch.load(f"vocoder/ckpt/{config['model_name']}", map_location=device)

    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder

