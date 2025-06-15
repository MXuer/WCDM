import h5py
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

class WSCMDataset(Dataset):
    def __init__(self, data_dir, norm=False):
        self.data, self.labels = self.read_h5_files(data_dir)
        self.norm = norm
        
    def read_h5_files(self, data_dir):
        h5_files = list(Path(data_dir).rglob('*.h5'))
        data = []
        label = []
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                data.extend(f['input'])
                label.extend(f['output'])
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