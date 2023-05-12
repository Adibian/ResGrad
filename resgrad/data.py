from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os

from .utils import normalize_residual, normalize_data

class SpectumDataset(Dataset):
    def __init__(self, config):
        super(SpectumDataset, self).__init__()
        self.config = config
        self.input_data_path = []
        self.target_data_path = []
        self.duration_data_path = []
        # i = 0
        for file_name in os.listdir(config['resgrade']['input_data_dir']):
            # i += 1
            # if i > 1000:
            #     break
            input_file_path = os.path.join(config['resgrade']['input_data_dir'], file_name)
            target_file_path = os.path.join(config['resgrade']['target_data_dir'], 'single_speaker-mel-' + file_name)
            duration_file_path = os.path.join(config['resgrade']['durations_dir'], 'single_speaker-duration-' + file_name)

            self.input_data_path.append(input_file_path)
            self.target_data_path.append(target_file_path)
            self.duration_data_path.append(duration_file_path)

        if config['resgrade']['model_type2'] == "segment-based":
            self.max_len = config['resgrade']['max_win_length']
            # self.win_size = config.window_size
        else:
            self.max_len = config['resgrade']['spectrum_max_size']
    
    def __getitem__(self, index):
        input_spec_path = self.input_data_path[index]
        input_spec = np.load(input_spec_path)
        target_spec_path = self.target_data_path[index]
        target_spec = np.load(target_spec_path)
        dutarions_path = self.duration_data_path[index]
        durations = np.load(dutarions_path)
        target_spec = torch.from_numpy(target_spec).T
        input_spec = torch.from_numpy(input_spec).squeeze()
        if self.config['resgrade']['normallize_spectrum']:
            input_spec = normalize_data(input_spec)
            target_spec = normalize_data(target_spec)

        if self.config['resgrade']['model_type2'] == "segment-based":
            start_phoneme_index = np.random.choice(len(durations)-4, 1)[0]
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
        print(input_spec.shape)
        print(target_spec.shape)
        print("###")
        input_spec = torch.nn.functional.pad(input_spec, (0, self.max_len-spec_size), mode = "constant", value = 0.0)
        target_spec = torch.nn.functional.pad(target_spec, (0, self.max_len-spec_size), mode = "constant", value = 0.0)
        
        residual_spec = target_spec - input_spec
        if self.config['resgrade']['normallize_residual']:
            residual_spec = normalize_residual(residual_spec)

        mask = torch.ones((1, input_spec.shape[-1]))
        mask[:,spec_size:] = 0

        if self.config['resgrade']['model_type1'] == "spec2residual":
            residual_spec = target_spec - input_spec
            if self.config.normallize_residual:
                residual_spec = normalize_residual(residual_spec)
            residual_spec = residual_spec*mask
            return input_spec, target_spec, residual_spec, mask
        else:
            return input_spec, target_spec, mask

    
    def __len__(self):
        return len(self.input_data_path)
    

def create_dataset(config):
    dataset = SpectumDataset(config)
    val_dataset, train_dataset = torch.utils.data.random_split(dataset, [config['resgrade']['val_size'], len(dataset)-(config['resgrade']['val_size'])])
    return DataLoader(train_dataset, batch_size=config['resgrade']['batch_size'], shuffle=config['resgrade']['shuffle_data']), \
                DataLoader(val_dataset, batch_size=config['resgrade']['batch_size'], shuffle=config['resgrade']['shuffle_data'])
