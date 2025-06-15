import h5py
import os
from tqdm import tqdm

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

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



if __name__=="__main__":
    dataset = WSCMDataset('data/test')
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for item in dataloader:
        inp, outp = item
        print(inp.size(), outp.size())
        break