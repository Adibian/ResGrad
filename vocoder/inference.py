import torch
import numpy as np

def infer(model, mel_spectrum, max_wav_value):
    if len(mel_spectrum.shape) == 2:
        mel_spectrum = mel_spectrum.unsqueeze(0)
    with torch.no_grad():
        wav = model(mel_spectrum).squeeze(1)[0]
    wav = wav.cpu().numpy() * max_wav_value * 0.97
    wav = wav.astype("int16")
    return wav