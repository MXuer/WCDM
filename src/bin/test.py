import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataset.dataset import WSCMDataset
from src.model.cnn import SignalCNN


def get_args():
    parser = argparse.ArgumentParser(description='测试WCDM模型')
    parser.add_argument('--test_dir', type=str, default='data/test', help='测试数据目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='模型路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='测试设备')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')
    return parser.parse_args()


def calculate_metrics(outputs, targets, threshold=0.5):
    """计算模型性能指标"""
    # 将输出二值化
    predictions = (outputs > threshold).float()
    
    # 计算准确率
    correct = (predictions == targets).float()
    accuracy = correct.mean().item()
    
    # 计算位错误率 (BER)
    bit_errors = (predictions != targets).float().mean().item()
    
    return {
        'accuracy': accuracy,
        'bit_error_rate': bit_errors
    }


def test(args):
    # 加载测试数据集
    print(f"加载测试数据集: {args.test_dir}")
    test_dataset = WSCMDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 加载模型
    model = SignalCNN(input_channels=4, output_dim=160)
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    # 定义损失函数
    criterion = nn.BCELoss()
    
    # 测试循环
    test_loss = 0.0
    all_metrics = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device).float()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 计算指标
            batch_metrics = calculate_metrics(outputs, targets, args.threshold)
            all_metrics.append(batch_metrics)
            
            # 更新损失
            test_loss += loss.item() * inputs.size(0)
    
    # 计算平均损失和指标
    test_loss /= len(test_dataset)
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    
    # 打印结果
    print(f"测试损失: {test_loss:.4f}")
    print(f"准确率: {avg_metrics['accuracy']:.4f}")
    print(f"位错误率: {avg_metrics['bit_error_rate']:.4f}")
    
    return test_loss, avg_metrics


if __name__ == "__main__":
    args = get_args()
    test_loss, metrics = test(args)