import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class Whole_Dataset(Dataset):
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
            receive_signal_data = np.array(f['receive_signal_data']).T
            receive_signal_pilot = np.array(f['receive_signal_pilot']).T
            output = np.array(f['original_bit'])

        receive_signal_data = torch.from_numpy(receive_signal_data).float().permute(2, 1, 0)
        receive_signal_pilot = torch.from_numpy(receive_signal_pilot).float().permute(2, 1, 0)
        return receive_signal_data, receive_signal_pilot, output


if __name__=="__main__":
    data_dir = "/data/duhu/WCDM/data_stages/rician_channel/fraction_delay/SF16_test_dataSet_160Bit_HDF520250708_092954/snr_-8dB"
    cset = D_Dataset(data_dir)
    dataloader = DataLoader(cset, batch_size=1, shuffle=False)
    for data_inp, pilot_inp, output in tqdm(dataloader):
        print(data_inp.shape, pilot_inp.shape, output.shape)
        break