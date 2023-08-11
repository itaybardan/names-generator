import os

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from names_generator import CONFIG
import logging

class NamesDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.target_sample_rate = 100
        self.frequencies = set()
        for root, dirs, files in os.walk(path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                waveform, sample_rate = torchaudio.load(file_path)
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
                # ensure if the audio is mono
                if waveform.shape[0] == 1:
                    for i in range(waveform.shape[1]):
                        self.frequencies.add(waveform[0][i].item())
        self.frequencies = sorted(self.frequencies)
        self.vocab_size = len(self.frequencies)
        self.freq_to_index = {frequency: i for i, frequency in enumerate(self.frequencies)}
        self.index_to_freq = {i: frequency for i, frequency in enumerate(self.frequencies)}
        self.waveforms = []
        for root, dirs, files in os.walk(path):
            for file_name in files:
                logging.info(f'loading {file_name}')
                file_path = os.path.join(root, file_name)
                waveform, sample_rate = torchaudio.load(file_path)
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
                # ensure if the audio is mono
                if waveform.shape[0] == 1:
                    waveform = waveform.flatten()
                    # TODO: maybe segmentation will be helpful
                    # normalized_waveform = waveform / torch.max(torch.abs(waveform))
                    # convert to frequency index
                    waveform = torch.tensor([self.freq_to_index[frequency.item()] for frequency in waveform])
                    self.waveforms.append(waveform)
                else:
                    logging.info(f'{file_path} is not mono, ignored')

        self.train_data = self.waveforms[:int(len(self.waveforms) * CONFIG.split)]
        self.val_data = self.waveforms[int(len(self.waveforms) * CONFIG.split):]

        logging.info(f'vocab size: {self.vocab_size}')

    def get_batch(self, mode, context_length, device, batch_size):
        # generate a small batch of data of inputs x and targets y
        if mode == 'train':
            waveform = self.train_data[torch.randint(len(self.train_data), (1,))]
        else:
            waveform = self.val_data[torch.randint(len(self.val_data), (1,))]
        random_times = torch.randint(len(waveform) - context_length, (batch_size,))
        x = torch.stack([waveform[random_time:random_time + context_length]
                         for random_time in random_times])
        y = torch.stack([waveform[random_time + 1:random_time + context_length + 1]
                         for random_time in random_times])
        x, y = x.to(device), y.to(device)
        return x, y

    def __getitem__(self, index):
        return self.waveforms[index]
