import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from librosa.util import normalize
from scipy.io.wavfile import read

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speakers = os.listdir(in_dir)
    for i, speaker in enumerate(speakers):
        loop =tqdm(os.listdir(os.path.join(in_dir, speaker)))
        loop.set_description(f'speaker count = {i+1}/{len(speakers)}')
        for file_name in loop:
            if file_name[-4:] != ".wav":
                continue
            base_name = file_name[:-4]
            text_path = os.path.join(
                in_dir, speaker, "{}.txt".format(base_name)
            )
            wav_path = os.path.join(
                in_dir, speaker, "{}.wav".format(base_name)
            )
            with open(text_path) as f:
                text = f.readline().strip("\n")
            text = _clean_text(text, cleaners)

            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            wav, _ = librosa.load(wav_path, sr=sampling_rate)
            wav = wav / max_wav_value
            wav = normalize(wav) * 0.95
            
            wavfile.write(
                os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.float32),
            )
            with open(
                os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                "w",
            ) as f1:
                f1.write(text)
