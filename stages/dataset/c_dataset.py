import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class C_Dataset(Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()
        self.h5_files = list(Path(data_dir).glob('*.h5'))
        print(f'loading {len(self.h5_files)} H5 files...')
        
    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, index):
        # channel_estimates (3, 2)
        # finger_data_channel_signal (160, 3, 2)
        with h5py.File(self.h5_files[index], 'r') as f:
            channel_estimates = np.array(f['channel_estimates']).T
            finger_data_channel_signal = np.array(f['finger_data_channel_signal']).T
            output = np.array(f['original_bit'])
        # expand channel_estimates from (3, 2) to (10, 3, 2)
        # channel_estimates = np.broadcast_to(channel_estimates[np.newaxis, ...], (1, 3, 2))
        # concatenate finger_data_channel_signal and channel_estimates
        # input = np.concatenate([finger_data_channel_signal, channel_estimates], axis=0)
        input = finger_data_channel_signal + channel_estimates
        input = torch.from_numpy(input).float().permute(1, 0, 2)
        
        return input, output


if __name__=="__main__":
    data_dir = "/data/duhu/WCDM/data_stages/rician_channel/fraction_delay/SF16_test_dataSet_160Bit_HDF520250708_092954/snr_-8dB"
    cset = C_Dataset(data_dir)
    dataloader = DataLoader(cset, batch_size=16, shuffle=True)
    for input, output in tqdm(dataloader):
        print(input.shape, output.shape)
        break
