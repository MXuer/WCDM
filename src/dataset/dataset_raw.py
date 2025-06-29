import h5py
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

class WSCMDataset(Dataset):
    def __init__(self, data_dir, norm=False):
        self.data, self.labels = self.read_h5_files(Path(data_dir))
        self.norm = norm
        
    def read_h5_files(self, data_dir):
        mutiPath_train_data_file = data_dir / 'mutiPath_train_data.h5'
        frame_verify_data_file = data_dir / 'frame_verify_data.h5'
        
        with h5py.File(mutiPath_train_data_file, 'r') as f:
            data = np.array(f['mutiPath_train_data']).T
        
        with h5py.File(frame_verify_data_file, 'r') as f:
            label = np.array(f['frame_verify_data']).T
            
        return data, label

    def __len__(self):
        return len(self.data)
    
    # 在dataset.py中添加数据归一化
    def __getitem__(self, idx):
        # 获取数据并转换为float32
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        data = data.permute(2, 0, 1)
        if self.norm:
            data = (data - data.mean()) / (data.std() + 1e-6)
        return data, label

if __name__=="__main__":
    dataset = WSCMDataset('wh_dataset/task1_mutipath_signal/fraction_delay/SF16_dataSet_fraction_test_160Bit_HDF5_fraction_delay20250621_134727/SNR_-6.0')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    x = batch[0]
    print("Input shape:", x.shape)
    print("Mean:", x.mean().item(), "Std:", x.std().item())
    print("Max:", x.max().item(), "Min:", x.min().item())