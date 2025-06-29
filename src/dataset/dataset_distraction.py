import h5py
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

class WSCMDataset(Dataset):
    def __init__(self, data_dir, norm=False):
        self.distraction_file = "spreading_screamble_uprate_result.h5"
        
        self.data, self.labels = self.read_h5_files(data_dir)
        self.norm = norm
        
    def read_h5_files(self, data_dir):
        with h5py.File(self.distraction_file, 'r') as f:
            distract_data = np.array(f['signal']).T
            distract_data = distract_data[None, :, :, None]
            distract_data = np.broadcast_to(distract_data, (1, 2, 10240, 3))
            
        h5_files = list(Path(data_dir).rglob('*.h5'))
        data = []
        label = []
        for h5_file in tqdm(h5_files, desc='reading h5 files...'):
            with h5py.File(h5_file, 'r') as f:
                each = np.array(f['input'])
                add_data = np.broadcast_to(distract_data, (each.shape[0], 2, 10240, 3))
                each = np.concatenate([each, add_data], axis=1)
                data.append(each)
                label.extend(f['output'])
        data = np.vstack(data)
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
    dataset = WSCMDataset('data/test')
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    batch = next(iter(dataloader))
    x = batch[0]
    print("Input shape:", x.shape)
    print("Mean:", x.mean().item(), "Std:", x.std().item())
    print("Max:", x.max().item(), "Min:", x.min().item())