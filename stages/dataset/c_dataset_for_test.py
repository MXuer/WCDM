import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class C_Dataset_forTest(Dataset):
    def __init__(self, data_dir, debug=False) -> None:
        super().__init__()
        self.h5_files = list(Path(data_dir).glob('*.h5'))
        self.debug_mode = debug
        print(f'加载 {len(self.h5_files)} 个H5文件...')
        
    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, index):
        # === 1. 加载原始数据 ===
        with h5py.File(self.h5_files[index], 'r') as f:
            channel_estimates = np.array(f['channel_estimates'])  # (2,3)
            finger_data_channel_signal = np.array(f['finger_data_channel_signal'])  # (2,3,160)
            # output = np.array(f['original_bit'])  # (160,)
            output = np.array(f['rake_output'])  # (160,)
            
            if self.debug_mode and index == 0:
                print("\n" + "="*50)
                print(f"文件: {self.h5_files[index].name}")
                print("实际原始数据维度:")
                print(f"信道估计(channel_estimates): {channel_estimates.shape}")
                print(f"时序信号(finger_data_channel_signal): {finger_data_channel_signal.shape}")
                print(f"输出标签(original_bit): {output.shape}")

        # === 2. 调整维度顺序 ===
        channel_estimates = channel_estimates.T  # (3,2)
        finger_data_channel_signal = finger_data_channel_signal.T  # (160,3,2)
        
        if self.debug_mode and index == 0:
            print("\n调整后维度:")
            print(f"信道估计: {channel_estimates.shape} (天线数×特征数)")
            print(f"时序信号: {finger_data_channel_signal.shape} (时间步×天线数×特征数)")

        # === 3. 处理信道估计数据 ===
        tensor_ce = torch.from_numpy(channel_estimates)  # (3,2)
        unsqueezed_ce = tensor_ce.unsqueeze(0)  # (1,3,2)
        expanded_ce = unsqueezed_ce.expand(160, -1, -1)  # (160,3,2)
        
        if self.debug_mode and index == 0:
            print("\n信道估计处理:")
            print(f"时间维度扩展: {expanded_ce.shape}")

        # === 4. 处理时序信号数据 ===
        tensor_signal = torch.from_numpy(finger_data_channel_signal)  # (160,3,2)
        
        if self.debug_mode and index == 0:
            print(f"时序信号张量: {tensor_signal.shape}")

        # === 5. 特征拼接 ===
        combined = torch.cat([tensor_signal, expanded_ce], dim=2)  # (160,3,4)
        
        if self.debug_mode and index == 0:
            print("\n特征拼接:")
            print(f"拼接后特征维度: {combined.shape}")

        # === 6. 维度重排 ===
        # 关键修正：将维度转换为 (天线数, 特征数, 时间步长)
        final_input = combined.permute(1, 0, 2).float()  # (3,4,160)
        
        if self.debug_mode and index == 0:
            print("\n维度重排:")
            print(f"最终输入维度: {final_input.shape}")
            print("物理意义: 天线数(3)×特征数(4)×时间步长(160)")

        return final_input, output


if __name__ == "__main__":
    data_dir = "/data/duhu/WCDM/data_stages/rician_channel/fraction_delay/SF16_test_dataSet_160Bit_HDF520250708_092954/snr_-8dB"
    
    # 创建数据集实例 (开启调试模式)
    cset = C_Dataset_forTest(data_dir, debug=True)
    
    # 创建数据加载器
    dataloader = DataLoader(cset, batch_size=16, shuffle=True)
    
    # 遍历数据加载器
    for batch_idx, (inputs, outputs) in enumerate(tqdm(dataloader)):
        if batch_idx == 0:
            print("\n" + "*"*50)
            print("批量数据维度:")
            print(f"输入数据形状: {inputs.shape} -> [批次(16), 天线(3), 特征(4), 时间步(160)]")
            print(f"输出标签形状: {outputs.shape} -> [批次(16), 序列长度(160)]")
            print("*"*50)
        
        # 为了节省时间，只看第一个批次
        break
