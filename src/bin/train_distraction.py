import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataset.dataset_distraction import WSCMDataset
from src.model.cnn_distraction import WCDMACNNDISTRACT
from src.loss.loss import CombinedLoss

torch.manual_seed(42)

def get_args():
    parser = argparse.ArgumentParser(description='训练WCDM模型')
    parser.add_argument('--data_dir', type=str, default='/data/duhu/WCDM/data_fraction_delay/train', help='训练数据目录')
    parser.add_argument('--test_dir', type=str, default='data/test', help='测试数据目录')
    parser.add_argument('--batch_size', type=int, default=512, help='批大小')
    parser.add_argument('--epochs', type=int, default=400, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--val_ratio', type=float, default=0.05, help='验证集比例')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='预热轮数')
    parser.add_argument('--log_dir', type=str, default='logs_fraction_delay/100k-add-distraction', help='TensorBoard日志目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints_fraction_delay/100k-add-distraction', help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--model-type', type=str, default='cnn', help='模型类别')
    return parser.parse_args()


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    """创建带有预热的学习率调度器"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性预热
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # 余弦退火
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    # 创建日志和保存目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    

    data_dir = Path(args.data_dir) / args.model_type

    args.log_dir = Path(args.log_dir) / f'{data_dir.name}'
    args.save_dir = Path(args.save_dir) / f'{data_dir.name}'

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 加载数据集
    print(f"加载训练数据集: {args.data_dir}")
    train_dataset = WSCMDataset(args.data_dir)
    
    # 划分训练集和验证集
    val_size = int(len(train_dataset) * args.val_ratio)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 初始化模型
    if args.model_type == "cnn":
        model = WCDMACNNDISTRACT()
    print(model)
    model = model.to(args.device)
    
    # 定义损失函数和优化器
    criterion = CombinedLoss()  # 二元交叉熵损失，适用于0/1输出
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器（带预热）
    scheduler = get_lr_scheduler(optimizer, args.warmup_epochs, args.epochs)
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses, val_losses, lrs = [], [], []
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for inputs, targets in train_pbar:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device).float()  # 确保目标是浮点型
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 更新统计
            train_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= train_size
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs = inputs.to(args.device)
                targets = targets.to(args.device).float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        
        val_loss /= val_size
        

        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(current_lr)

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
        log_file = os.path.join(args.save_dir, f'epoch_{epoch}.json')
        log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'current_lr': current_lr
            }
        with open(log_file, 'w') as f:
            f.write(json.dumps(log_dict, ensure_ascii=False, indent=2))

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    writer.close()
    print("训练完成！")


    # 绘制训练损失曲线
    plt.figure(figsize=(12, 8))
    
    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(val_losses, 'r-', linewidth=2, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 学习率曲线
    plt.subplot(2, 1, 2)
    plt.plot(lrs, 'g-', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(args.save_dir / 'training_progress.png', dpi=300)
    print('训练过程曲线已保存为 training_progress.png')
    
    # 绘制损失曲线 (仅损失)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(val_losses, 'r-', linewidth=2, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(args.save_dir / 'loss_curve.png', dpi=300)
    print('损失曲线已保存为 loss_curve.png')


if __name__ == "__main__":
    args = get_args()
    train(args)