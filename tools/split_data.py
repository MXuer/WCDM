import h5py
import numpy as np
import random
from pathlib import Path


def write2dataset(dataset, output_file):
    x_arr, y_arr = [], []
    for x, y in dataset:
        x_arr.append(x)
        y_arr.append(y)
    x_arr, y_arr = np.array(x_arr), np.array(y_arr)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('input', data=x_arr)
        f.create_dataset('output', data=y_arr)

def main():
    data_dir = Path("SF16_dataSet_160Bit_HDF5_20250613_203530")
    exp_dir = Path("SF16_dataSet_160Bit_HDF5_20250613_203530_split")
    train_dir = exp_dir / "train"
    test_dir = exp_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_rate = 0.1
    for each in data_dir.glob("*"):
        print(each)
        input_file = each / 'mutiPath_train_data.h5'
        output_file = each / 'frame_verify_data.h5'
        with h5py.File(input_file, 'r') as f:
            input = np.array(f['mutiPath_train_data']).T
        with h5py.File(output_file, 'r') as f:
            output = np.array(f['frame_verify_data']).T

        dataset = []
        for index in range(2000):
            x, y = input[index, :, :, :], output[index]
            dataset.append([x, y])

        random.shuffle(dataset)
        test_num = int(len(dataset) * test_rate)
        testdata = dataset[:test_num]
        traindata = dataset[test_num:]
        write2dataset(traindata, train_dir / f'{each.name}.h5')
        write2dataset(testdata, test_dir / f'{each.name}.h5')

if __name__ == '__main__':
    main()