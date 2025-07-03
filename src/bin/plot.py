import json
import argparse
import matplotlib.pyplot as pltfrom pathlib import Path


def main(args):
    train_losses, val_losses, lrs = [], [], []
    checkpoint_dir = Path(args.checkpoint_dir)
    json_files = list(checkpoint_dir.glob('*.json'))
    # 按照epoch排序
    json_files = sorted(list(Path('.').glob('*.json')), key=lambda x:int(x.stem.replace('epoch_', '')))
    
    for json_file in json_files:
        data = json.load(open(json_file))
        train_losses.append(data['train_loss'])
        val_losses.append(data['val_loss'])
        lrs.append(data['current_lr'])

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
    plt.savefig(checkpoint_dir / 'training_progress.png', dpi=300)
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
    plt.savefig(checkpoint_dir / 'loss_curve.png', dpi=300)
    print('损失曲线已保存为 loss_curve.png')



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='画图')
    parser.add_argument('--checkpoint_dir', type=str, help='模型存储目录')
    args = parser.parse_args()
    main(args)