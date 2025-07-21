import argparse
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tqdm import tqdm
from stages.dataset.whole import Whole_Dataset
from torch.utils.data import DataLoader
from stages.model.D_model import DUNET
from stages.model.P_model import PUNET

def get_args():
    parser = argparse.ArgumentParser(description='测试WCDM模型')
    parser.add_argument('--test_dir', type=str, default='/data/duhu/WCDM/data_stages/rayleigh_channel/fraction_delay/SF16_test_dataSet_160Bit_HDF520250714_145955', help='测试数据目录')
    parser.add_argument('--batch_size', type=int, default=512, help='批大小')
    parser.add_argument('--data-model', type=str, default='ckpt_stages_mix_pad_perbit/dunet/best_model.pth', help='模型路径')
    parser.add_argument('--pilot-model', type=str, default='checkpoints_stage_mix/punet/best_model.pth', help='模型路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='测试设备')
    parser.add_argument('--threshold', type=float, default=0, help='二值化阈值')
    parser.add_argument('--model-type', type=str, default='cunet', help='模型类别')
    return parser.parse_args()

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main():
    args = get_args()
    device = args.device

    dmodel = DUNET().to(device)
    pmodel = PUNET().to(device)

    dmodel = load_checkpoint(dmodel, args.data_model, device)
    pmodel = load_checkpoint(pmodel, args.pilot_model, device)
    test_dirs = [ele for ele in list(Path(args.test_dir).glob('*')) if ele.is_dir()]
    test_dirs = sorted(test_dirs, key=lambda x:int(x.name.split('_')[-1][:-2]))
    with open(Path(args.test_dir) / 'result.txt', 'w') as f:
        for data_dir in test_dirs:
            if not data_dir.is_dir(): continue
            print(f"Processing directory: {data_dir}")
            dataset = Whole_Dataset(data_dir)
            total_error = 0
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            for batch_idx, (data_inp, pilot_inp, output) in enumerate(tqdm(dataloader)):
                data_inp = data_inp.to(device)
                pilot_inp = pilot_inp.to(device)
                output = output.detach().cpu().numpy()
                bs = data_inp.shape[0]
                combined_data, combined_pilot = [], []
                for ii in range(3):
                    # 数据模型预测
                    data_out = dmodel(data_inp[:, ii, :, :].unsqueeze(1)).detach().cpu().numpy()
                    # 导频模型预测
                    pilot_out = pmodel(pilot_inp[:, ii, :, :].unsqueeze(1)).detach().cpu().numpy()
                    combined_data.append(data_out)
                    combined_pilot.append(pilot_out)
                for b_index in range(bs):
                    data_c1, data_c2, data_c3 = combined_data[0][b_index], combined_data[1][b_index], combined_data[2][b_index]
                    pilot_c1, pilot_c2, pilot_c3 = combined_pilot[0][b_index], combined_pilot[1][b_index], combined_pilot[2][b_index]

                    data_c1_complex = data_c1[0, :] + 1j * data_c1[1, :]
                    data_c2_complex = data_c2[0, :] + 1j * data_c2[1, :]
                    data_c3_complex = data_c3[0, :] + 1j * data_c3[1, :]

                    pilot_c1_complex = pilot_c1[0] + 1j * pilot_c1[1]
                    pilot_c2_complex = pilot_c2[0] + 1j * pilot_c2[1]
                    pilot_c3_complex = pilot_c3[0] + 1j * pilot_c3[1]

                    mrc_data = np.conj(pilot_c1_complex) * data_c1_complex + np.conj(pilot_c2_complex) * data_c2_complex + np.conj(pilot_c3_complex) * data_c3_complex
                    # print("MRC Data:", mrc_data.shape)

                    # 合并多径: 沿路径维度求和
                    combined_signal = mrc_data

                    # 3. 硬判决 (取实部)
                    # 由于信号是复数，我们取实部进行判决
                    signal_real = np.real(combined_signal)
                    # print("Signal Real:", signal_real)
                    hard_decision = np.where(signal_real >= 0, 1, 0)
                    output_bs = output[b_index]
                    error = 0
                    for index in range(160):
                        if hard_decision[index] != output_bs[index]:
                            error += 1
                    total_error += error

            print(f"Total Errors in batch {data_dir}: {total_error} | {total_error / len(dataset) / 160}")
            f.write(f"Total Errors in batch {data_dir}: {total_error} | {total_error / len(dataset) / 160}\n")

if __name__ == "__main__":
    main()
