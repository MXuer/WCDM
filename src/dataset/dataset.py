import h5py
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from glob import glob

class WSCMDataset(Dataset):
    def __init__(self, data_dir):
        self.data, self.labels = self.read_h5_files(data_dir)
        
    def read_h5_files(self, data_dir):
        h5_files = glob(os.path.join(data_dir, '*.h5'))
        data = []
        label = []
        for h5_file in tqdm(h5_files):
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
        
        # 对输入数据进行归一化（如果数据范围不是[0,1]）
        data = (data - data.mean()) / (data.std() + 1e-8)  # 避免除零错误
        
        return data, label

if __name__=="__main__":
    dataset = WSCMDataset('data/test')
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for item in dataloader:
        inp, outp = item
        print(inp.size(), outp.size())
        break