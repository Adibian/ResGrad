import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from librosa.util import normalize

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    speaker = "single_speaker"
    with open(os.path.join(in_dir, "train.txt"), encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            parts = line.strip().split("|")
            base_name = parts[0]
            text = parts[2]
            wav_path = os.path.join(in_dir, 'wavs', base_name + '.wav')

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
                