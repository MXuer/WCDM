import h5py
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader


class P_Dataset(Dataset):
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
        return len(self.h5_files) * 3

    def __getitem__(self, index):
        # channel_estimates (3, 2)
        # finger_data_channel_signal (160, 3, 2)
        file_index = index // 3
        data_index = index % 3
        with h5py.File(self.h5_files[file_index], 'r') as f:
            receive_signal_data = np.array(f['receive_signal_pilot']).T
            channel_estimates = np.array(f['channel_estimates']).T
        receive_signal_data = torch.from_numpy(receive_signal_data).float().permute(2, 1, 0)[data_index, :, :].unsqueeze(0)
        channel_estimates = torch.from_numpy(channel_estimates)[data_index, :]
        return receive_signal_data, channel_estimates


if __name__=="__main__":
    data_dir = "/data/duhu/WCDM/data_stages/rician_channel/fraction_delay/SF16_test_dataSet_160Bit_HDF520250708_092954/snr_-8dB"
    data_dir = [
                    'data_stages/rayleigh_channel/fraction_delay/SF16_test_dataSet_160Bit_HDF520250714_145955',
                    'data_stages/rayleigh_channel/integer_delay/SF16_test_dataSet_160Bit_HDF520250716_085214',
                    'data_stages/rayleigh_channel_8SF/fraction_delay/SF16_test_dataSet_160Bit_HDF5_8SF20250716_230127',
                    'data_stages/rayleigh_channel_8SF/integer_delay/SF16_test_dataSet_160Bit_HDF520250716_230330',
                    'data_stages/rician_channel/fraction_delay/SF16_test_dataSet_160Bit_HDF520250708_092954',
                    'data_stages/rician_channel/integer_delay/SF16_test_dataSet_160Bit_HDF520250709_125145',
                    'data_stages/rician_channel_8SF/fraction_delay/SF16_test_dataSet_160Bit_HDF520250714_231619',
                    'data_stages/rician_channel_8SF/integer_delay/SF16_test_dataSet_160Bit_HDF520250716_085430'
                ]
    cset = P_Dataset(data_dir, val=True)
    print(len(cset))
    # dataloader = DataLoader(cset, batch_size=1, shuffle=True)
    # for input, output in tqdm(dataloader):
    #     print(input.shape, output.shape)
    #     break