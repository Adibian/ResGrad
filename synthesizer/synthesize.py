import time
import torch
import yaml
import numpy as np

from .utils.tools import to_device, prepare_outputs
from .text import text_to_sequence


def synthesize(model, batch, control_values, preprocess_config, device, p_target=None, d_target=None, e_target=None):
    pitch_control, energy_control, duration_control = control_values

    batch = to_device(batch, device)
    with torch.no_grad():
        # Forward
        output = model(
            *(batch[1:]),            
             
            p_targets=p_target,
            e_targets=e_target,
            d_targets=d_target,

            p_control=pitch_control,
            e_control=energy_control,
            d_control=duration_control
        )
        mel, durations, pitch, energy = prepare_outputs(
            batch,
            output,
            preprocess_config,
        )
    return mel[0].to(device), durations[0].to(device), pitch[0].to(device), energy[0].to(device)

def infer(model, grapheme, control_values, preprocess_config, device, speaker=None, p_target=None, d_target=None, e_target=None):
    t = str(time.time()).replace('.', '_')
    ids = [t]
    if speaker is None:
        speakers = np.array([0])
    else:
        speakers = np.array([speaker])
    texts = np.array([text_to_sequence(grapheme, preprocess_config['preprocessing']['text']['text_cleaners'])])
    text_lens = np.array([len(texts[0])])
    batch = (ids, speakers, texts, text_lens, max(text_lens))
    model.eval()
    mel, durations, pitch, energy = synthesize(model, batch, control_values, preprocess_config, device, p_target, d_target, e_target)
    return mel, durations, pitch, energy

