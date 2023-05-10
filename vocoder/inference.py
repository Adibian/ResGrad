import torch

def infer(model, mel_spectrum, max_wav_value):
    with torch.no_grad():
        wav = model(mel_spectrum).squeeze(1)

    wav = wav.cpu().numpy() * max_wav_value * 0.97
    wav = wav.astype("int16")
    return wav