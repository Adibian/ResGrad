from utils import load_models, load_yaml_file
from synthesizer.synthesize import infer as synthesizer_infer

import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import yaml


def read_input_data(raw_data_path):
    input_texts = {}
    for speaker in os.listdir(raw_data_path):
        dir_path = os.path.join(raw_data_path, speaker)
        for file_name in os.listdir(dir_path):
            if '.lab' in file_name:
                file_path = os.path.join(dir_path, file_name)
                with open(file_path) as f:
                    input_text = f.read()
                    input_texts[(speaker, file_name.replace(".lab", ""))] = input_text.strip()
    return input_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesizer_restore_step", type=int, required=True)
    parser.add_argument("--raw_data_path", type=str, default="synthesizer/raw_data", required=False)
    parser.add_argument("-c", "--config", type=str, default='config/Persian/config.yaml', required=False, help="path to config.yaml")
    args = parser.parse_args()
    
    # Read Config
    config = load_yaml_file(args.config)

    restore_steps = {"synthesizer":args.synthesizer_restore_step, "resgrad":None, "vocoder":None}
    synthesizer_model, _, _ = load_models(restore_steps, config['synthesizer'])
    text_data = read_input_data(args.raw_data_path)

    current_path = os.getcwd()
    duration_dir = os.path.join(current_path, config['resgrad']['data']['durations_dir'])
    mel_pred_dir = os.path.join(current_path, config['resgrad']['data']['input_data_dir'])
    mel_target_dir = os.path.join(current_path, config['resgrad']['data']['mel_target'])

    os.makedirs(mel_pred_dir, exist_ok=True)
    if not os.path.islink(mel_target_dir):
        os.symlink(os.path.join(current_path, config['synthesizer']['preprocess']['path']['preprocessed_path'], 'mel'), mel_target_dir, target_is_directory=True)
    if not os.path.islink(duration_dir):
        os.symlink(os.path.join(current_path, config['synthesizer']['preprocess']['path']['preprocessed_path'], 'duration'), duration_dir, target_is_directory=True)

    device = config['synthesizer']['main']['device']
    for (speaker, file_name), text in tqdm(text_data.items()):
        dur_file_name = speaker + "-duration-" + file_name + ".npy"
        dur_target = torch.from_numpy(np.load(os.path.join(duration_dir, dur_file_name))).to(device).unsqueeze(0)

        control_values = 1.0,1.0,1.0
        mel_prediction, _, _, _ = synthesizer_infer(synthesizer_model, text, control_values, config, 
                                                    device, d_target=dur_target)
        file_path = os.path.join(mel_pred_dir, file_name)
        np.save(file_path, mel_prediction[0].cpu())


if __name__ == "__main__":
    main()
# python resgrad_data.py --synthesizer_restore_step 240
