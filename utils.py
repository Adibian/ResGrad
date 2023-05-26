from synthesizer.utils.model import get_model as load_synthesizer_model
from resgrad.utils import load_model as load_resgrad_model
from vocoder.utils import get_vocoder

from synthesizer.utils.tools import plot_mel

import os
import json
from scipy.io import wavfile
from matplotlib import pyplot as plt
import yaml
import time

def save_result(mel_prediction, wav, pitch, energy, preprocess_config, result_dir, file_name):
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]
    fig = plot_mel([(mel_prediction.cpu().numpy(), pitch.cpu().numpy(), energy.cpu().numpy())], stats, ["Synthetized Spectrogram"])
    plt.savefig(os.path.join(result_dir, "{}.png".format(file_name)))
    plt.close()

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(os.path.join(result_dir, "{}.wav".format(file_name)), sampling_rate, wav)


def load_models(all_restore_step, config):
    synthesizer_model, resgrad_model, vocoder_model = None, None, None
    device = config['synthesizer']['main']['device']
    if all_restore_step['synthesizer'] not in [None, 0]:
        synthesizer_model = load_synthesizer_model(all_restore_step['synthesizer'], config).to(device)
    if all_restore_step['vocoder'] not in [None, 0]:
        vocoder_model = get_vocoder(all_restore_step['vocoder']).to(device)
    device = config['resgrad']['main']['device']
    if all_restore_step['resgrad'] not in [None, 0]:
        resgrad_model = load_resgrad_model(config['resgrad'], train=False, restore_model_step=all_restore_step['resgrad']).to(device)
    return synthesizer_model, resgrad_model, vocoder_model

def load_yaml_file(path):
    ## define custom tag handler
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return str(os.path.join(*[str(i) for i in seq]))
    
    ## register the tag handler
    yaml.add_constructor('!join', join)
    data = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    return data

def get_file_name(args):
    file_name_parts = []
    if args.result_file_name:
        file_name_parts.append(args.result_file_name)
    if args.speaker_id:
        file_name_parts.append("spk" + str(args.speaker_id))
    if len(file_name_parts) == 0:
       file_name_parts.append(str(time.time()).replace('.', '_'))
    file_name_parts.append("FastSpeech")
    file_name = "_".join(file_name_parts)
    return file_name