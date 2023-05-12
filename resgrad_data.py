from utils import load_models, load_yaml_file
from synthesizer.synthesize import infer as synthesizer_infer

import argparse
import os
import csv
from tqdm import tqdm
import numpy as np
import torch


def read_input_data(raw_data_path):
    input_texts = {}
    speakers = os.listdir(raw_data_path)
    for i, speaker in enumerate(speakers):
        dir_path = os.path.join(raw_data_path, speaker)
        # loop = tqdm(os.listdir(dir_path))
        # loop.set_description(f'speaker count = {i+1}/{len(speakers)}')
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
    parser.add_argument("--raw_data_path", type=str, default="synthesizer/raw_data/Persian", required=False)
    parser.add_argument("--config", type=str, default='config/Persian/config.yaml', required=False, help="path to config.yaml")
    args = parser.parse_args()
    
    # Read Config
    config = load_yaml_file(args.config)

    restore_steps = {"synthesizer":args.synthesizer_restore_step, "resgrad":None, "vocoder":None}
    synthesizer_model, _, _ = load_models(restore_steps, config)
    print("Load input data...")
    text_data = read_input_data(args.raw_data_path)
    print("{} text input is loaded.".format(len(text_data)))

    # duration_dir = os.path.join(current_path, config['resgrad']['data']['durations_dir'])
    mel_pred_dir = os.path.join(config['resgrad']['data']['input_mel_dir'])
    # mel_target_dir = os.path.join(current_path, config['resgrad']['data']['target_data_dir'])

    os.makedirs(mel_pred_dir, exist_ok=True)
    # if not os.path.islink(mel_target_dir):
    #     os.symlink(os.path.join(current_path, config['synthesizer']['preprocess']['path']['preprocessed_path'], 'mel'), mel_target_dir, target_is_directory=True)
    # if not os.path.islink(duration_dir):
    #     os.symlink(os.path.join(current_path, config['synthesizer']['preprocess']['path']['preprocessed_path'], 'duration'), duration_dir, target_is_directory=True)

    resgrad_data = []
    device = config['synthesizer']['main']['device']
    # i = 0
    for (speaker, file_name), text in tqdm(text_data.items()):
        # i +=1 
        # if i>30:
        #     break
        dur_file_name = speaker + "-duration-" + file_name + ".npy"
        dur_path = os.path.join(config['synthesizer']['preprocess']['path']['preprocessed_path'], 'duration', dur_file_name)

        mel_target_file_name = speaker + "-mel-" + file_name + ".npy"
        mel_target_path = os.path.join(config['synthesizer']['preprocess']['path']['preprocessed_path'], 'mel', mel_target_file_name)

        ### Synthersize mel-spectrum and save as data for resgrad
        dur_target = torch.from_numpy(np.load(dur_path)).to(device).unsqueeze(0)
        control_values = 1.0,1.0,1.0
        mel_prediction, _, _, _ = synthesizer_infer(synthesizer_model, text, control_values, config['synthesizer']['preprocess'], 
                                                    device, d_target=dur_target)
        mel_pred_path = os.path.join(mel_pred_dir, speaker + "-pred_mel-" + file_name + ".npy")
        np.save(mel_pred_path, mel_prediction.cpu())

        resgrad_data.append({'speaker': speaker, 'predicted_mel':mel_pred_path, 'target_mel':mel_target_path, 'duration':dur_path})

    with open(config['resgrad']['data']['metadata_path'], 'w') as file: 
        fields = ['speaker', 'predicted_mel', 'target_mel', 'duration']
        writer = csv.DictWriter(file, fieldnames = fields)
        writer.writeheader() 
        writer.writerows(resgrad_data)


if __name__ == "__main__":
    main()
# python resgrad_data.py --synthesizer_restore_step 240
