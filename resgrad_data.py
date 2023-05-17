from utils import load_models, load_yaml_file
from synthesizer.synthesize import infer as synthesizer_infer

import argparse
import os
import csv
from tqdm import tqdm
import numpy as np
import torch
import json

def read_input_data(data_file_path):
    input_texts = {}
    with open(data_file_path, mode='r') as f:
        lines = f.readlines()
    for line in lines:
        fields = line.split("|")
        file_name, speaker, input_text = fields[0], fields[1], fields[2]
        input_texts[(speaker, file_name)] = input_text.strip()
    return input_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesizer_restore_step", type=int, required=True)
    parser.add_argument("--data_file_path", type=str, default="dataset/Persian/synthesizer_data/train.txt", required=False)
    parser.add_argument("--config", type=str, default='config/Persian/config.yaml', required=False, help="path to config.yaml")
    args = parser.parse_args()
    
    # Read Config
    config = load_yaml_file(args.config)

    restore_steps = {"synthesizer":args.synthesizer_restore_step, "resgrad":None, "vocoder":None}
    synthesizer_model, _, _ = load_models(restore_steps, config)
    print("Load input data...")
    text_data = read_input_data(args.data_file_path)
    print("{} inputs data is loaded.".format(len(text_data)))

    with open(os.path.join(config['synthesizer']['preprocess']['path']['preprocessed_path'], "speakers.json")) as f:
        speaker_map = json.load(f)

    mel_pred_dir = os.path.join(config['resgrad']['data']['input_mel_dir'])
    os.makedirs(mel_pred_dir, exist_ok=True)

    resgrad_data = []
    device = config['synthesizer']['main']['device']
    # i = 0
    for (speaker, file_name), text in tqdm(text_data.items()):
        # i +=1 
        # if i>30:
        #     break
        dur_file_name = speaker + "-duration-" + file_name + ".npy"
        pitch_file_name = speaker + "-pitch-" + file_name + ".npy"
        dur_path = os.path.join(config['synthesizer']['preprocess']['path']['preprocessed_path'], 'duration', dur_file_name)
        pitch_path = os.path.join(config['synthesizer']['preprocess']['path']['preprocessed_path'], 'pitch', pitch_file_name)

        mel_target_file_name = speaker + "-mel-" + file_name + ".npy"
        mel_target_path = os.path.join(config['synthesizer']['preprocess']['path']['preprocessed_path'], 'mel', mel_target_file_name)

        ### Synthersize mel-spectrum and save as data for resgrad
        dur_target = torch.from_numpy(np.load(dur_path)).to(device).unsqueeze(0)
        pitch_target = torch.from_numpy(np.load(pitch_path)).to(device).unsqueeze(0)
        control_values = 1.0,1.0,1.0
        mel_prediction, _, _, _ = synthesizer_infer(synthesizer_model, text, control_values, config['synthesizer']['preprocess'], 
                                                    device, speaker=speaker_map[speaker], d_target=dur_target, p_target=pitch_target)

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
