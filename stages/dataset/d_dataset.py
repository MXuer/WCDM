import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class D_Dataset(Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()
        self.h5_files = list(Path(data_dir).glob('*.h5'))
        print(f'loading {len(self.h5_files)} H5 files...')
        
    def __len__(self):
        return len(self.h5_files) * 3

    def __getitem__(self, index):
        # channel_estimates (3, 2)
        # finger_data_channel_signal (160, 3, 2)
        file_index = index // 3
        data_index = index % 3
        with h5py.File(self.h5_files[file_index], 'r') as f:
            receive_signal_data = np.array(f['receive_signal_data']).T
            finger_data_channel_signal = np.array(f['finger_data_channel_signal']).T
        receive_signal_data = torch.from_numpy(receive_signal_data).float().permute(2, 1, 0)[data_index, :, :].unsqueeze(0)
        finger_data_channel_signal = torch.from_numpy(finger_data_channel_signal).permute(1, 2, 0)[data_index, :, :]
        return receive_signal_data, finger_data_channel_signal


if __name__=="__main__":
    data_dir = "/data/duhu/WCDM/data_stages/rician_channel/fraction_delay/SF16_test_dataSet_160Bit_HDF520250708_092954/snr_-8dB"
    cset = D_Dataset(data_dir)
    dataloader = DataLoader(cset, batch_size=1, shuffle=True)
    for input, output in tqdm(dataloader):
        print(input.shape, output.shape)
        break