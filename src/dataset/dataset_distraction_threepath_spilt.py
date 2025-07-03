import h5py
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class WSCMDataset(Dataset):
    def __init__(self, data_dir, norm=False, chunk_size=64):
        self.distraction_file = "/data/duhu/WCDM/raw_data/spreading_screamble_uprate_result.h5"
        self.data, self.labels = self.read_h5_files(data_dir)
        self.norm = norm
        self.chunk_size = chunk_size
        
        # 计算每个样本可以分成多少块（基于时间维度）
        self.chunks_per_sample = self.data.shape[2] // chunk_size
        
        self.total_samples = self.data.shape[0] * self.chunks_per_sample
        print(f"原始样本数: {self.data.shape[0]}, 每样本块数: {self.chunks_per_sample}, 总块数: {self.total_samples}")
        print(f"标签数据形状: {self.labels.shape}")
        print(f"主数据形状: {self.data.shape}") 
        
    def read_h5_files(self, data_dir):
        # 搜索两个特定的.h5文件
        data_file = None
        label_file = None
        
        for path in Path(data_dir).rglob('*.h5'):
            if path.name == "mutiPath_train_data.h5":
                data_file = path
            elif path.name == "frame_verify_data.h5":
                label_file = path
        
        if data_file is None:
            raise FileNotFoundError(f"未找到 mutiPath_train_data.h5 文件 ({data_dir})")
        if label_file is None:
            raise FileNotFoundError(f"未找到 frame_verify_data.h5 文件 ({data_dir})")
        
        # 加载扰码数据
        with h5py.File(self.distraction_file, 'r') as f:
            distract_data = np.array(f['signal']).T
            print(f"扰码数据原始维度: {distract_data.shape}")
            if distract_data.shape != (2, 10240):
                raise ValueError(f"扰码数据维度不符合要求: 应该是(2, 10240)，实际是{distract_data.shape}")
        
        # 读取主数据文件并进行选择
        with h5py.File(data_file, 'r') as f:
            data = np.array(f['mutiPath_train_data'])
            print(f"主数据原始维度: {data.shape}")
            
            # 转置并调整形状: (样本数, 通道数, 时间步数, 特征通道数)
            data = data.transpose(3, 2, 1, 0)  # 调整为(样本数, 4, 10240, 3)
            
            # 关键修改: 只选择第二维的第一行和第三行(索引0和2)
            data = data[:, [1, 3], :, :]  # 现在形状为(样本数, 2, 10240, 3)
            print(f"选择特定通道后数据形状: {data.shape}")
        
        n_samples = data.shape[0]
        
        # 扩展扰码数据 (2, 10240) -> (样本数, 2, 10240, 3)
        add_data = np.tile(distract_data[np.newaxis, :, :, np.newaxis], 
                           (n_samples, 1, 1, data.shape[3]))
        print(f"扰码数据扩展后形状: {add_data.shape}")
        
        # 沿通道维度连接主数据和扰码数据
        combined_data = np.concatenate([data, add_data], axis=1)
        print(f"组合数据形状: {combined_data.shape}")  # 预期: (样本数, 4, 10240, 3)
        
        # 读取标签文件
        with h5py.File(label_file, 'r') as f:
            labels = np.array(f['frame_verify_data'])
            print(f"标签数据原始维度: {labels.shape}")
            labels = labels.T
            print(f"标签数据转置后形状: {labels.shape}")
        
        return combined_data, labels
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # 计算原始样本索引和块索引
        sample_idx = idx // self.chunks_per_sample
        chunk_idx = idx % self.chunks_per_sample
        
        # 获取原始样本数据 (4, 10240, 3)
        sample_data = self.data[sample_idx]
        
        # 计算块的起始和结束位置
        start = chunk_idx * self.chunk_size
        end = start + self.chunk_size
        
        # 切取数据块 (4, 块大小, 3)
        chunk_data = sample_data[:, start:end, :]

        # 调整维度顺序: (4, 64, 3) -> (3, 4, 64)
        chunk_data = np.transpose(chunk_data, (2, 0, 1))
        
        # 确保样本索引在有效范围内
        if sample_idx >= len(self.labels):
            raise IndexError(f"样本索引 {sample_idx} 超出标签数组范围 {len(self.labels)}")
        
        # 获取对应标签
        label_vector = self.labels[sample_idx]
        
        # 确保块索引在有效范围内
        if chunk_idx >= len(label_vector):
            raise IndexError(f"块索引 {chunk_idx} 超出标签向量范围 {len(label_vector)}")
        
        # 获取当前块对应的标签
        label = label_vector[chunk_idx]
        
        # 转换为Tensor
        data_tensor = torch.tensor(chunk_data, dtype=torch.float32)
        
        # 归一化处理
        if self.norm:
            # 对每个特征通道单独归一化
            for c in range(data_tensor.shape[0]):
                channel_data = data_tensor[c]
                mean = channel_data.mean()
                std = channel_data.std()
                if std > 1e-7:  # 避免除以0
                    data_tensor[c] = (channel_data - mean) / std
        
        return data_tensor, torch.tensor([label], dtype=torch.float32)

if __name__ == "__main__":
    try:
        dataset = WSCMDataset('/data/duhu/WCDM/raw_data/SF16_dataSet_fraction_test_160Bit_HDF5_fraction_delay20250621_134727/SNR_-6.0', norm=True)
        print(f"数据集大小: {len(dataset)}")
        
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        
        # 获取第一个batch
        batch = next(iter(dataloader))
        x, y = batch
        print("\n第一个batch信息:")
        print(f"输入形状: {x.shape}")  # 预期 (100, 3, 4, 64)
        print(f"输出形状: {y.shape}")  # 预期 (100, 1)
        
        # 验证数据内容
        print("\n验证数据内容...")
        for i in range(5):
            sample, label = dataset[i]
            print(f"样本 {i}: 数据形状 {sample.shape}, 标签 {label.item()}")
        
        # 获取所有数据块并进行验证
        print("\n遍历数据集并统计...")
        all_x = []
        all_y = []
        for x_batch, y_batch in tqdm(dataloader, desc="处理数据"):
            all_x.append(x_batch)
            all_y.append(y_batch)
        
        # 拼接所有数据
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        
        print(f"\n最终数据统计:")
        print(f"输入数据总大小: {all_x.shape}")
        print(f"输出数据总大小: {all_y.shape}")
        
        # 检查索引是否正确
        # print("\n验证样本索引和块索引:")
        # for i in range(0, len(dataset), max(1, len(dataset)//10)):
        #     sample_idx = i // dataset.chunks_per_sample
        #     chunk_idx = i % dataset.chunks_per_sample
        #     print(f"索引 {i}: 样本 {sample_idx}, 块 {chunk_idx}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()