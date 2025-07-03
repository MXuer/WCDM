import h5py
import numpy as np
import random
from pathlib import Path


def write2dataset(dataset, output_file: Path):
    output_file.parent.mkdir(exist_ok=True, parents=True)
    x_arr, y_arr = [], []
    for x, y in dataset:
        x_arr.append(x)
        y_arr.append(y)
    x_arr, y_arr = np.array(x_arr), np.array(y_arr)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('input', data=x_arr)
        f.create_dataset('output', data=y_arr)

def main():
    data_dir = Path("/data/duhu/WCDM/wh_dataset/task1_mutipath_signal/OnePath/onePath_SF16_Test_dataSet_160Bit_HDF520250626_234059")
    exp_dir = Path("data_1path_clean")
    train_dir = exp_dir / "test"
    # test_dir = exp_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    # test_dir.mkdir(parents=True, exist_ok=True)
    
    test_rate = 0
    for each in data_dir.glob("*"):
        print(each)
        input_file = each / 'mutiPath_train_data.h5'
        output_file = each / 'frame_verify_data.h5'
        print(f'loading {input_file}...')
        with h5py.File(input_file, 'r') as f:
            input = np.array(f['mutiPath_train_data']).T
        print(f'loading {output_file}...')
        with h5py.File(output_file, 'r') as f:
            output = np.array(f['frame_verify_data']).T

        dataset = []
        for index in range(input.shape[0]):
            x, y = input[index, :, :], output[index]
            dataset.append([x, y])
        h5_index = 0
        if test_rate == 0:
            for index in range(0, len(dataset), 1000):
                print(f'writing {index} => {index+1000}')
                write2dataset(dataset[index:index+1000], train_dir/ each.name / f'{h5_index}.h5')
                h5_index += 1
        else:
            random.shuffle(dataset)
            test_num = int(len(dataset) * test_rate)
            testdata = dataset[:test_num]
            traindata = dataset[test_num:]
            write2dataset(traindata, train_dir / f'{each.name}.h5')
            # write2dataset(testdata, test_dir / f'{each.name}.h5')

if __name__ == '__main__':
    main()