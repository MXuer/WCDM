import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataset.dataset_distraction_threepath_spilt import WSCMDataset
from src.model.cnn_distraction_threepath_split import WCDMACNNDISTRACT
from src.model.delay_unet_p3 import DelayAwareUNet

def get_args():
    parser = argparse.ArgumentParser(description='测试WCDM模型')
    parser.add_argument('--test_dir', type=str, default='/data/duhu/WCDM/raw_data/Integer_delay/New_SF16_dataSet_160Bit_HDF5_20250615_022354', help='测试数据目录')
    parser.add_argument('--batch_size', type=int, default=160 * 80, help='批大小')
    parser.add_argument('--model_path', type=str, default='checkpoints_Integer_delay/100k-add-distraction/threepath/cnn/best_model.pth', help='模型路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='测试设备')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')
    parser.add_argument('--model-type', type=str, default='cnn', help='模型类别')
    return parser.parse_args()


def calculate_metrics(outputs, targets, threshold=0.5):
    """计算模型性能指标"""
    # 将输出二值化
    predictions = (outputs >= threshold).float()
    
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
    # 一次性加载模型（放在循环外部）
    if args.model_type == "cnn":
        model = WCDMACNNDISTRACT()
    elif args.model_type == "unet":
        model = DelayAwareUNet()
    
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    # 收集所有结果
    results = []
    
    # 遍历所有子文件夹
    sub_test_dirs = [d for d in Path(args.test_dir).iterdir() if d.is_dir()]
    for sub_dir in sub_test_dirs:
        test_dataset = WSCMDataset(sub_dir)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        criterion = nn.BCELoss()
        test_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"测试 {sub_dir.name}"):
                inputs = inputs.to(args.device)
                targets = targets.to(args.device).float()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                batch_metrics = calculate_metrics(outputs, targets, args.threshold)
                all_metrics.append(batch_metrics)
                test_loss += loss.item() * inputs.size(0)
        
        # 计算当前目录的指标
        test_loss /= len(test_dataset)
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        
        # 记录结果而不是返回
        results.append((str(sub_dir), test_loss, avg_metrics))
        print(f"{sub_dir.name}\t{test_loss:.4f}\t{avg_metrics['accuracy']:.4f}")
    
    # 最终返回所有结果（或直接在这里处理结果）
    return results


if __name__ == "__main__":
    args = get_args()
    all_results = test(args)
    
    # 打印汇总结果
    print("\n测试结果汇总:")
    all_results = sorted(all_results, key=lambda x:float(x[0].split('-')[-1]))
    for path, loss, metrics in all_results:
        print(f"目录: {Path(path).name}\t损失: {loss:.4f}\t准确率: {metrics['accuracy']:.6f}\t误码率: {metrics['bit_error_rate']:.6f}")
        # print(f"  损失: {loss:.4f}")
        # print(f"  准确率: {metrics['accuracy']:.6f}")
        # print(f"  误码率: {metrics['bit_error_rate']:.6f}")