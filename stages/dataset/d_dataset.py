import h5py
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader


class D_Dataset(Dataset):
    def __init__(self, data_dir, val=False) -> None:
        super().__init__()
        self.h5_files = []
        if val == False:
            if type(data_dir) == str:
                self.h5_files = list(Path(data_dir).rglob('*.h5'))
            elif type(data_dir) == list:
                for dir in data_dir:
                    print(f'loading {dir}...')
                    self.h5_files += list(Path(dir).rglob('*.h5'))
            random.shuffle(self.h5_files)
            self.h5_files = self.h5_files[:200000]
            print(f'loading {len(self.h5_files)} H5 files...')
        else:
            h5_files = []
            for dir in data_dir:
                h5_files += list(Path(dir).rglob('*.h5'))
            name2h5_files = defaultdict(list)
            for h5_file in tqdm(h5_files):
                name = f'{h5_file.parent.parent.stem}--{h5_file.parent.stem}'
                name2h5_files[name].append(h5_file)
            self.h5_files = []
            for name, h5_files in name2h5_files.items():
                self.h5_files += random.sample(h5_files, 100)
            print(f'loading {len(self.h5_files)} H5 files...')
        
    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, index):
        # channel_estimates (3, 2)
        # finger_data_channel_signal (160, 3, 2)
        with h5py.File(self.h5_files[index], 'r') as f:
            receive_signal_data = np.array(f['receive_signal_data']).T
            finger_data_channel_signal = np.array(f['finger_data_channel_signal']).T
        receive_signal_data = torch.from_numpy(receive_signal_data).float().permute(2, 1, 0)
        finger_data_channel_signal = torch.from_numpy(finger_data_channel_signal).permute(1, 2, 0)
        return receive_signal_data, finger_data_channel_signal


if __name__=="__main__":
    data_dir = "/data/duhu/WCDM/data_stages/rician_channel/fraction_delay/SF16_test_dataSet_160Bit_HDF520250708_092954/snr_-8dB"
    cset = D_Dataset(data_dir)
    dataloader = DataLoader(cset, batch_size=4, shuffle=False)
    for input, output in tqdm(dataloader):
        inp = input.view(input.shape[0]*input.shape[1], input.shape[2], input.shape[3]).unsqueeze(1)
        out = output.view(output.shape[0]*output.shape[1], output.shape[2], output.shape[3])
        # inp = input.view(input.shape[0]*input.shape[1], -1)
        print(input.shape, output.shape)
        print(inp.shape, out.shape)
        break