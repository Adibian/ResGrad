from synthesizer.utils.model import get_model as load_synthesizer_model
from resgrad.utils import load_model as load_resgrad_model
from vocoder.utils import get_vocoder

from synthesizer.utils.tools import plot_mel

import os
import json
import time
from scipy.io import wavfile
from matplotlib import pyplot as plt
import yaml

def save_result(mel_prediction, wav, pitch, energy, preprocess_config, result_dir):
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]
    fig = plot_mel([(mel_prediction.cpu().numpy(), pitch, energy)], stats, ["Synthetized Spectrogram"])
    file_name = str(time.time()).replace('.', '_')
    plt.savefig(os.path.join(result_dir, "{}.png".format(file_name)))
    plt.close()

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(os.path.join(result_dir, "{}.wav".format(file_name)), sampling_rate, wav)


def load_models(all_restore_step, synthesizer_configs):
    synthesizer_model, resgrad_model, vocoder_model = None, None, None
    if all_restore_step['synthesizer'] is not None:
        synthesizer_model = load_synthesizer_model(all_restore_step['synthesizer'], (synthesizer_configs['preprocess_config'], \
                                        synthesizer_configs['model_config'], synthesizer_configs['train_config']))
    if all_restore_step['resgrad'] is not None:
        resgrad_model = load_resgrad_model(train=False, restore_model_epoch=all_restore_step['regrad'])
    if all_restore_step['vocoder'] is not None:
        vocoder_model = get_vocoder(all_restore_step['vocoder'])
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