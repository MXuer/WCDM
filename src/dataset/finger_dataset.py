import h5py
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

class WSCMFingerDataset(Dataset):
    def __init__(self, data_dir, norm=False):
        self.data, self.labels = self.read_h5_files(data_dir)
        self.norm = norm
        
    def read_h5_files(self, data_dir):
        input_file = Path(data_dir) / 'finger_signal_add_channel_estimate_data.h5'
        label_file = Path(data_dir) / 'frame_verify_data.h5'
        data = []
        label = []
        with h5py.File(input_file, 'r') as f:
            data = np.array(f['finger_signal_dataset']).T
        with h5py.File(label_file, 'r') as f:
            label = np.array(f['frame_verify_data']).T
        return data, label

    def __len__(self):
        return len(self.data)
    # 在dataset.py中添加数据归一化
    def __getitem__(self, idx):
        # 获取数据并转换为float32
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        data = data.permute(0, 2, 1)
        if self.norm:
            data = (data - data.mean()) / (data.std() + 1e-6)
        return data, label

if __name__=="__main__":
    dataset = WSCMFingerDataset('data_easy/test/SNR_-6.0')
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    batch = next(iter(dataloader))
    x = batch[0]
    print("Input shape:", x.shape)
    print("Mean:", x.mean().item(), "Std:", x.std().item())
    print("Max:", x.max().item(), "Min:", x.min().item())