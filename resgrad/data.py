from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import csv
import json

from .utils import normalize_residual, normalize_data

class SpectumDataset(Dataset):
    def __init__(self, config):
        super(SpectumDataset, self).__init__()
        self.config = config
        with open(config['data']['speaker_map_path']) as f:
            self.speaker_map = json.load(f)

        self.input_data_path = []
        self.target_data_path = []
        self.duration_data_path = []
        self.speakers = []
        with open(config['data']['metadata_path'], mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                line_count += 1
                if line_count > 1:
                    self.input_data_path.append(row['predicted_mel'])
                    self.target_data_path.append(row['target_mel'])
                    self.duration_data_path.append(row['duration'])
                    self.speakers.append(self.speaker_map[row['speaker']])

        if config['model']['model_type2'] == "segment-based":
            self.max_len = config['data']['max_win_length']
        else:
            self.max_len = config['data']['spectrum_max_size']
    
    def __getitem__(self, index):
        input_spec_path = self.input_data_path[index]
        input_spec = np.load(input_spec_path)
        target_spec_path = self.target_data_path[index]
        target_spec = np.load(target_spec_path)
        dutarions_path = self.duration_data_path[index]
        durations = np.load(dutarions_path)
        target_spec = torch.from_numpy(target_spec).T
        input_spec = torch.from_numpy(input_spec).squeeze()
        if self.config['data']['normallize_spectrum']:
            input_spec = normalize_data(input_spec, self.config)
            target_spec = normalize_data(target_spec, self.config)

        if self.config['model']['model_type2'] == "segment-based":
            start_phoneme_index = np.random.choice(len(durations)-min(4, len(durations)-1), 1)[0]
            end_phoneme_index = 0
            for i in range(start_phoneme_index+1, len(durations)+1):
                win_length = sum(durations[start_phoneme_index:i])
                if win_length > self.max_len:
                    end_phoneme_index = i-1
                    break
            if end_phoneme_index == 0:
                end_phoneme_index = len(durations)
                for i in range(start_phoneme_index):
                    start_phoneme_index -= 1
                    win_length = sum(durations[start_phoneme_index:end_phoneme_index])
                    if win_length > self.max_len:
                        start_phoneme_index += 1
                        break
            win_start = sum(durations[:start_phoneme_index])
            win_end = sum(durations[:end_phoneme_index])

            input_spec = input_spec[:,win_start:win_end]
            target_spec = target_spec[:,win_start:win_end]
        
        spec_size = input_spec.shape[-1]
        input_spec = torch.nn.functional.pad(input_spec, (0, self.max_len-spec_size), mode = "constant", value = 0.0)
        target_spec = torch.nn.functional.pad(target_spec, (0, self.max_len-spec_size), mode = "constant", value = 0.0)
        
        residual_spec = target_spec - input_spec
        if self.config['data']['normallize_residual']:
            residual_spec = normalize_residual(residual_spec, self.config)

        mask = torch.ones((1, input_spec.shape[-1]))
        mask[:,spec_size:] = 0

        speaker = self.speakers[index]

        if self.config['model']['model_type1'] == "spec2residual":
            residual_spec = target_spec - input_spec
            if self.config['data']['normallize_residual']:
                residual_spec = normalize_residual(residual_spec, self.config)
            residual_spec = residual_spec*mask
            return input_spec, target_spec, residual_spec, mask, speaker
        else:
            return input_spec, target_spec, mask, speaker

    
    def __len__(self):
        return len(self.input_data_path)
    

def create_dataset(config):
    dataset = SpectumDataset(config)
    val_dataset, train_dataset = torch.utils.data.random_split(dataset, [config['data']['val_size'], len(dataset)-(config['data']['val_size'])])
    return DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=config['data']['shuffle_data']), \
                DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=config['data']['shuffle_data'])
