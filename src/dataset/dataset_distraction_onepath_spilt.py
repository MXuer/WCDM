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
        self.chunks_per_sample = self.data.shape[2] // chunk_size
        self.total_samples = self.data.shape[0] * self.chunks_per_sample
        print(f"原始样本数: {self.data.shape[0]}, 每样本块数: {self.chunks_per_sample}, 总块数: {self.total_samples}")
        print(f"标签数据形状: {self.labels.shape}")  # 添加标签形状打印
        
    def read_h5_files(self, data_dir):
        # 搜索两个特定的.h5文件
        data_file = None
        label_file = None
        
        # 在data_dir及其子目录中递归搜索目标文件
        for path in Path(data_dir).rglob('*.h5'):
            if path.name == "mutiPath_train_data.h5":
                data_file = path
            elif path.name == "frame_verify_data.h5":
                label_file = path
        
        # 检查是否找到文件
        if data_file is None:
            raise FileNotFoundError(f"未找到 mutiPath_train_data.h5 文件 ({data_dir})")
        if label_file is None:
            raise FileNotFoundError(f"未找到 frame_verify_data.h5 文件 ({data_dir})")
        
        # 加载扰码数据 (维度应该是 2x10240)
        with h5py.File(self.distraction_file, 'r') as f:
            distract_data = np.array(f['signal']).T  # 转置后维度为 2×10240
            print(f"扰码数据原始维度: {distract_data.shape}")
            # 确保维度正确
            if distract_data.shape != (2, 10240):
                raise ValueError(f"扰码数据维度不符合要求: 应该是(2, 10240)，实际是{distract_data.shape}")
        
        # 读取主数据文件
        with h5py.File(data_file, 'r') as f:
            data = np.array(f['mutiPath_train_data'])
            print(f"主数据原始维度: {data.shape}")
            # 转置并调整形状: (样本数, 通道数, 时间步数)
            # 原始形状: (3, 10240, 4, 样本数)
            # 转置后: (样本数, 4, 10240, 3)
            data = data.transpose(3, 2, 1, 0)  # 调整为(样本数, 4, 10240, 3)
            data = data[..., 0]  # 取第一个通道: (样本数, 4, 10240)
        
        # 为整个数据集创建扰码数据的扩展
        n_samples = data.shape[0]
        
        # 扩展扰码数据 (2, 10240) -> (样本数, 2, 10240)
        add_data = np.tile(distract_data[np.newaxis, :, :], (n_samples, 1, 1))
        
        # 沿通道维度连接主数据和扰码数据
        combined_data = np.concatenate([data, add_data], axis=1)
        print(f"组合数据形状: {combined_data.shape}")
        
        # 读取标签文件
        with h5py.File(label_file, 'r') as f:
            labels = np.array(f['frame_verify_data'])
            print(f"标签数据原始维度: {labels.shape}")
            
            # 修正关键错误：转置标签数据以匹配样本数量
            # 原始形状: (160, 样本数) 应转置为 (样本数, 160)

            labels = labels.T  # 转置为 (样本数, 160)
            print(f"标签数据转置后形状: {labels.shape}")
        
        return combined_data, labels
    

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # 计算原始样本索引和块索引
        sample_idx = idx // self.chunks_per_sample
        chunk_idx = idx % self.chunks_per_sample
        
        # 获取原始样本数据 (通道数, 10240)
        sample_data = self.data[sample_idx]  # (6, 10240)
        
        # 计算块的起始和结束位置
        start = chunk_idx * self.chunk_size
        end = start + self.chunk_size
        
        # 切取数据块 (通道数, 块大小)
        chunk_data = sample_data[:, start:end]

        # 关键修复：添加通道维度
        chunk_data = chunk_data[np.newaxis, :, :]  # 现在形状为 (1, 6, 64)
        
        # 确保样本索引在有效范围内
        if sample_idx >= len(self.labels):
            raise IndexError(f"样本索引 {sample_idx} 超出标签数组范围 {len(self.labels)}")
        
        # 获取对应标签 (整个样本的160个标签)
        label_vector = self.labels[sample_idx]  # (160,)
        
        # 确保块索引在有效范围内
        if chunk_idx >= len(label_vector):
            raise IndexError(f"块索引 {chunk_idx} 超出标签向量范围 {len(label_vector)}")
        
        # 获取当前块对应的标签 (标量)
        label = label_vector[chunk_idx]
        
        # 转换为Tensor
        data_tensor = torch.tensor(chunk_data, dtype=torch.float32)
        
        # 归一化处理 (可选项)
        if self.norm:
            # 这里是对整个数据块归一化
            mean = data_tensor.mean()
            std = data_tensor.std()
            if std > 1e-7:  # 避免除以0
                data_tensor = (data_tensor - mean) / std
        
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
        print(f"输入形状: {x.shape}")  # 应该是 (100, 6, 64)
        print(f"输出形状: {y.shape}")  # 应该是 (100, 1)
        
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
        print("\n验证样本索引和块索引:")
        for i in range(0, len(dataset), len(dataset)//10):
            sample_idx = i // dataset.chunks_per_sample
            chunk_idx = i % dataset.chunks_per_sample
            print(f"索引 {i}: 样本 {sample_idx}, 块 {chunk_idx}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
