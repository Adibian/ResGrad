import time
import torch
import numpy as np
from string import punctuation
import re
from g2p_en import G2p

from .utils.tools import to_device, prepare_outputs
from .text import text_to_sequence

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return np.array(sequence)


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

def infer(model, text, control_values, preprocess_config, device, speaker=None, p_target=None, d_target=None, e_target=None):
    t = str(time.time()).replace('.', '_')
    ids = [t]
    speakers = np.array([speaker])
    if preprocess_config["preprocessing"]["text"]["language"] == "fa":
        texts = np.array([text_to_sequence(text, preprocess_config['preprocessing']['text']['text_cleaners'])])
    elif preprocess_config["preprocessing"]["text"]["language"] == "en":
        texts = np.array([preprocess_english(text, preprocess_config)])
    text_lens = np.array([len(texts[0])])
    batch = (ids, speakers, texts, text_lens, max(text_lens))
    model.eval()
    mel, durations, pitch, energy = synthesize(model, batch, control_values, preprocess_config, device, p_target, d_target, e_target)
    return mel, durations, pitch, energy

