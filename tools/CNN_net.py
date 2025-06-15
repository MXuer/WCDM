import os
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc  # 在文件开头导入垃圾回收模块
from src.dataset.dataset import WSCMDataset

# 定义自定义损失函数（结合MSE和二元交叉熵）
class CombinedLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # MSE权重
        self.beta = beta    # BCE权重
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        
    def forward(self, predictions, targets):
        # 均方误差部分
        mse_loss = self.mse(predictions, targets)
        # 二元交叉熵部分
        bce_loss = self.bce(predictions, targets)
        # 加权组合
        return self.alpha * mse_loss + self.beta * bce_loss

# 定义CNN模型架构
# 修改后的CNN模型架构（添加Dropout和正则化）
class WCDMACNN(nn.Module):
    def __init__(self, input_height, input_width, input_channels, output_size):
        super(WCDMACNN, self).__init__()
        
        # 修改后的卷积层序列（添加Dropout）
        self.conv_layers = nn.Sequential(
            # 第一卷积层
            nn.Conv2d(input_channels, 64, kernel_size=(2, 64), stride=(2, 2), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第二卷积层
            nn.Conv2d(64, 128, kernel_size=(2, 16), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第三卷积层
            nn.Conv2d(128, 256, kernel_size=(1, 8), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加30%的Dropout
            
            # 第四卷积层
            nn.Conv2d(256, 128, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加20%的Dropout
            
            # 第五卷积层
            nn.Conv2d(128, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.LeakyReLU(0.1),
            
            # 第六卷积层
            nn.Conv2d(64, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # 展平后计算特征大小
        self._init_dense(input_height, input_width, input_channels, output_size)
        
    def _init_dense(self, input_height, input_width, input_channels, output_size):
        # 创建两个样本作为测试输入
        test_input = torch.zeros(2, input_channels, input_height, input_width)
        
        # 设置模型为评估模式进行测试
        original_mode = self.conv_layers.training
        self.conv_layers.eval()
        with torch.no_grad():
            conv_output = self.conv_layers(test_input)
        self.conv_layers.train(original_mode)
        
        flattened_size = conv_output.view(2, -1).shape[1]
        
        # 修改后的全连接层序列（添加更多Dropout）
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1000),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.Dropout(0.5),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x

def main():
    # 初始化计时
    import time
    start_time = time.time()
    
    # 系统参数
    SF_single = 16
    sampsPerSym = 4
    bit_num = 160
    frame_length = bit_num * SF_single
    sampsPerFrame = frame_length * sampsPerSym
    path_count = 3
    
    # # 可调参数
    # sample_ratio = 0.6
    # train_test_ratio = 0.5
    
    # # 数据集路径
    # dataset_path = 'SF16_dataSet_160Bit_HDF5_20250613_203530'
    # if not os.path.exists(dataset_path):
    #     print("找不到文件夹，脚本终止。")
    #     return
    
    # # 获取所有SNR文件夹
    # snr_folders = [f for f in os.listdir(dataset_path) if f.startswith('SNR_')]
    # print(f'找到 {len(snr_folders)} 个SNR文件夹')
    
    # # 按SNR分组存储数据
    # all_snrs = []
    # train_samples_by_snr = {}
    # train_labels_by_snr = {}
    # test_samples_by_snr = {}
    # test_labels_by_snr = {}
    
    # # 遍历SNR文件夹
    # for folder in snr_folders:
    #     folder_path = os.path.join(dataset_path, folder)
        
    #     # 提取SNR值
    #     try:
    #         snr_value = float(folder.split('_')[1])
    #     except:
    #         print(f"跳过无效文件夹: {folder}")
    #         continue
            
    #     all_snrs.append(snr_value)
    #     print(f'处理SNR={snr_value} dB的文件夹: {folder_path}')
        
    #     # 读取标签数据
    #     verify_file = os.path.join(folder_path, 'frame_verify_data.h5')
    #     if not os.path.exists(verify_file):
    #         print(f'文件夹 {folder_path} 中缺少验证数据文件，跳过')
    #         continue
            
    #     with h5py.File(verify_file, 'r') as f:
    #         # 打印原始标签数据维度
    #         labels_raw = np.array(f['frame_verify_data'])
    #         print(f'  原始标签数据维度 (MATLAB顺序): {labels_raw.shape}')
            
    #         # 注意：MATLAB保存的是列优先，Python读取时需要转置
    #         labels_data = labels_raw.T
    #         print(f'  转置后标签数据维度: {labels_data.shape}')
            
    #     # 读取训练数据
    #     mutipath_file = os.path.join(folder_path, 'mutiPath_train_data.h5')
    #     if not os.path.exists(mutipath_file):
    #         print(f'文件夹 {folder_path} 中缺少训练数据文件，跳过')
    #         continue
            
    #     with h5py.File(mutipath_file, 'r') as f:
    #         # MATLAB保存的顺序与Python不同，需要调整维度顺序
    #         mutiPath_raw = np.array(f['mutiPath_train_data'])
    #         print(f'  原始训练数据维度 (MATLAB顺序): {mutiPath_raw.shape}')
            
    #         # 重新排列维度为：(帧数, 4, 信号长度, 多径数)
    #         mutiPath_data = np.transpose(mutiPath_raw, (3, 2, 1, 0))
    #         print(f'  调整后训练数据维度: {mutiPath_data.shape}')
        
    #     # 样本数量
    #     num_frames = labels_data.shape[0]
    #     print(f'  找到 {num_frames} 个样本(帧)')
        
    #     # 抽样机制
    #     num_samples_to_use = max(1, int(num_frames * sample_ratio))
    #     if num_samples_to_use < num_frames:
    #         print(f'  抽取 {num_samples_to_use} 个样本使用 ({sample_ratio*100:.0f}%)')
    #         selected_indices = np.random.choice(num_frames, num_samples_to_use, replace=False)
    #     else:
    #         print(f'  使用全部 {num_frames} 个样本')
    #         selected_indices = np.arange(num_frames)
        
    #     # 划分训练集和测试集
    #     num_train = int(train_test_ratio * num_samples_to_use)
    #     train_indices = selected_indices[:num_train]
    #     test_indices = selected_indices[num_train:]
        
    #     # 提取训练数据和标签
    #     train_samples = mutiPath_data[train_indices]
    #     train_labels = labels_data[train_indices]
        
    #     # 提取测试数据和标签
    #     test_samples = mutiPath_data[test_indices]
    #     test_labels = labels_data[test_indices]
        
    #     print(f'  训练样本维度: {train_samples.shape}, 训练标签维度: {train_labels.shape}')
    #     print(f'  测试样本维度: {test_samples.shape}, 测试标签维度: {test_labels.shape}')
        
    #     # 存储到字典
    #     train_samples_by_snr[snr_value] = train_samples
    #     train_labels_by_snr[snr_value] = train_labels
    #     test_samples_by_snr[snr_value] = test_samples
    #     test_labels_by_snr[snr_value] = test_labels
        
    #     print(f'  SNR {snr_value} dB: {len(train_samples)}训练样本 + {len(test_samples)}验证样本')
    
    # # 合并所有SNR的数据
    # train_samples_list = []
    # train_labels_list = []
    # test_samples_list = []
    # test_labels_list = []
    
    # for snr in train_samples_by_snr:
    #     train_samples_list.append(train_samples_by_snr[snr])
    #     train_labels_list.append(train_labels_by_snr[snr])
        
    # for snr in test_samples_by_snr:
    #     test_samples_list.append(test_samples_by_snr[snr])
    #     test_labels_list.append(test_labels_by_snr[snr])
    
    # all_train_samples = np.vstack(train_samples_list)
    # all_train_labels = np.vstack(train_labels_list)
    # all_test_samples = np.vstack(test_samples_list)
    # all_test_labels = np.vstack(test_labels_list)


    # # # 注意这里交换了验证集和测试集=========================================================
    # # all_train_samples = np.vstack(test_samples_list)
    # # all_train_labels = np.vstack(test_labels_list)
    # # all_test_samples = np.vstack(train_samples_list)
    # # all_test_labels = np.vstack(train_labels_list)

    # # 立即释放不再需要的列表
    # del train_samples_list, train_labels_list, test_samples_list, test_labels_list
    # gc.collect()
    
    # print('\n合并后维度:')
    # print(f'训练样本维度: {all_train_samples.shape} (样本数, 高度=4, 宽度=10240, 通道=3)')
    # print(f'训练标签维度: {all_train_labels.shape} (样本数, 标签长度=160)')
    # print(f'测试样本维度: {all_test_samples.shape} (样本数, 高度=4, 宽度=10240, 通道=3)')
    # print(f'测试标签维度: {all_test_labels.shape} (样本数, 标签长度=160)')
    
    # # 维度转换: (帧数, 高度, 宽度, 通道) -> (帧数, 通道, 高度, 宽度) - PyTorch要求格式
    # all_train_samples = np.transpose(all_train_samples, (0, 3, 1, 2))
    # all_test_samples = np.transpose(all_test_samples, (0, 3, 1, 2))
    
    # # 验证维度
    # print('\n转换后维度(PyTorch格式):')
    # print(f'训练样本维度: {all_train_samples.shape} (样本数, 通道=3, 高度=4, 宽度=10240)')
    # print(f'测试样本维度: {all_test_samples.shape} (样本数, 通道=3, 高度=4, 宽度=10240)')
    
    # # 检查维度是否符合预期
    # expected_height = 4
    # expected_width = sampsPerFrame  # 10240
    # expected_channels = path_count  # 3
    
    # actual_shape = all_train_samples.shape
    # if (actual_shape[1] == expected_channels and 
    #     actual_shape[2] == expected_height and 
    #     actual_shape[3] == expected_width):
    #     print("\n✅ 维度验证: 输入数据维度符合预期")
    # else:
    #     print(f"\n❌ 维度异常: 实际维度{actual_shape[1:]} 预期维度({expected_channels}, {expected_height}, {expected_width})")
    
    # # 打印数据信息摘要
    # print(f"\n数据摘要:")
    # print(f"训练样本数量: {all_train_samples.shape[0]}")
    # print(f"测试样本数量: {all_test_samples.shape[0]}")
    # print(f"标签长度: {all_train_labels.shape[1]}")
    
    # # 打乱数据
    # train_indices = np.random.permutation(len(all_train_samples))
    # test_indices = np.random.permutation(len(all_test_samples))
    
    # all_train_samples = all_train_samples[train_indices]
    # all_train_labels = all_train_labels[train_indices]
    # all_test_samples = all_test_samples[test_indices]
    # all_test_labels = all_test_labels[test_indices]
    
    # data_load_time = time.time() - start_time
    # print(f'\n数据处理完成，耗时: {data_load_time/60:.2f} 分钟')
    
    # # 转换为PyTorch张量
    # train_samples_tensor = torch.tensor(all_train_samples, dtype=torch.float32)
    # train_labels_tensor = torch.tensor(all_train_labels, dtype=torch.float32)
    # test_samples_tensor = torch.tensor(all_test_samples, dtype=torch.float32)
    # test_labels_tensor = torch.tensor(all_test_labels, dtype=torch.float32)
    
    # # 创建数据集和数据加载器
    # batch_size = 10
    # train_dataset = TensorDataset(train_samples_tensor, train_labels_tensor)
    # test_dataset = TensorDataset(test_samples_tensor, test_labels_tensor)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    
    train_dataset = WSCMDataset('data/train')
    
    # 划分训练集和验证集
    val_size = int(len(train_dataset) * 0.05)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
    
    # 创建数据加载器
    batch_size = 1024
    train_loader = DataLoader(train_subset, batch_size=1024, shuffle=True, num_workers=4)
    test_loader = DataLoader(val_subset, batch_size=1024, shuffle=False, num_workers=4)
    print(f'\n数据加载器:')
    print(f'训练集: {len(train_dataset)}个样本, 每批{batch_size}个样本')
    # print(f'测试集: {len(test_dataset)}个样本, 每批{batch_size}个样本')
    
    # 初始化模型
    input_height, input_width = 4, 10240
    input_channels =3
    output_size = 160
    
    print(f'\n模型输入参数:')
    print(f'输入高度: {input_height}')
    print(f'输入宽度: {input_width}')
    print(f'输入通道数: {input_channels}')
    print(f'输出大小: {output_size}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    model = WCDMACNN(input_height, input_width, input_channels, output_size).to(device)
    
    # 打印模型架构
    print("\n模型架构:")
    print(model)
    
    # 计算可训练参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数总数: {total_params}")
    
    # ===== 关键验证点: 输入输出维度检查 =====
    print("\n===== 模型输入输出维度验证 =====")
    
    # 保存当前模型模式
    was_training = model.training
    
    # 设置为评估模式 (禁用BatchNorm的统计更新，避免小批量问题)
    model.eval()
    print(f"临时设置模型为评估模式 (禁用BatchNorm统计)")
    
    # # 测试随机输入数据 - 使用最小批次大小2
    # test_input = torch.randn(2, input_channels, input_height, input_width).to(device)
    # print(f"测试输入维度: {test_input.shape} (批次大小=2, 通道={input_channels}, 高度={input_height}, 宽度={input_width})")
    
    # # 禁用梯度计算
    # with torch.no_grad():
    #     test_output = model(test_input)
    
    # print(f"测试输出维度: {test_output.shape} (批次大小=2, 输出大小={output_size})")
    
    # 使用真实训练数据
    # sample_input, sample_label = next(iter(train_loader))
    # sample_input = sample_input.to(device)
    # sample_label = sample_label.to(device)
    # print(f"\n真实数据输入维度: {sample_input.shape} (批次大小={batch_size}, 通道={input_channels}, 高度={input_height}, 宽度={input_width})")
    # print(f"真实标签维度: {sample_label.shape} (批次大小={batch_size}, 标签长度={output_size})")
    
    # 禁用梯度计算
    # with torch.no_grad():
    #     real_output = model(sample_input)
    
    # print(f"模型输出维度: {real_output.shape} (批次大小={batch_size}, 标签长度={output_size})")
    
    # # 验证输出维度是否与标签匹配
    # if real_output.shape == sample_label.shape:
    #     print("\n✅ 维度一致性验证: 模型输出与标签维度匹配")
    # else:
    #     print(f"\n❌ 维度异常: 模型输出维度{real_output.shape} 标签维度{sample_label.shape}")
    
    # 使用验证数据
    # val_input, val_label = next(iter(test_loader))
    # val_input = val_input.to(device)
    # val_label = val_label.to(device)
    # print(f"\n验证数据输入维度: {val_input.shape} (批次大小={batch_size}, 通道={input_channels}, 高度={input_height}, 宽度={input_width})")
    # print(f"验证标签维度: {val_label.shape} (批次大小={batch_size}, 标签长度={output_size})")
    
    # 禁用梯度计算
    # with torch.no_grad():
    #     val_output = model(val_input)
    
    # print(f"验证输出维度: {val_output.shape} (批次大小={batch_size}, 标签长度={output_size})")
    
    # # 验证输出维度是否与验证标签匹配
    # if val_output.shape == val_label.shape:
    #     print("\n✅ 维度一致性验证: 验证输出与标签维度匹配")
    # else:
    #     print(f"\n❌ 维度异常: 验证输出维度{val_output.shape} 标签维度{val_label.shape}")
    
    # # 恢复模型原始模式
    # if was_training:
    #     model.train()
    #     print("\n恢复模型为训练模式")
    # else:
    #     print("\n模型保持为评估模式")
    
    # 定义损失函数、优化器和学习率调度器
    criterion = CombinedLoss(alpha=10, beta=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    
    # 训练参数
    num_epochs = 200
    best_val_loss = float('inf')
    patience = 100
    counter = 0
    
    # 初始化记录器
    train_loss_history = []
    val_loss_history = []
    learning_rates = []
    
    print("\n开始训练...")
    training_start = time.time()
    
    # 打印第一个batch的训练过程
    print("\n===== 训练过程维度监控 =====")
    print("监控第一个batch的训练过程维度:")
    
    # 确保模型处于训练模式
    model.train()
    print("设置模型为训练模式")
    
    # 获取第一个batch
    first_batch_inputs, first_batch_labels = next(iter(train_loader))
    first_batch_inputs = first_batch_inputs.to(device)
    first_batch_labels = first_batch_labels.to(device)
    
    # 梯度清零
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(first_batch_inputs)
    loss = criterion(outputs, first_batch_labels)
    
    print(f"训练输入维度: {first_batch_inputs.shape}")
    print(f"模型输出维度: {outputs.shape}")
    print(f"训练标签维度: {first_batch_labels.shape}")
    print(f"损失值: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 更新权重
    optimizer.step()
    
    for epoch in range(num_epochs):
        # ==== 训练阶段 ====
        model.train()
        epoch_train_loss = 0.0
        batch_count = 0
        
        # 使用tqdm创建进度条
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Training')
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 更新训练损失
            epoch_train_loss += loss.item() * inputs.size(0)
            batch_count += inputs.size(0)
            
            # 更新进度条描述
            train_loop.set_postfix(loss=loss.item())
        
        # 计算平均训练损失
        epoch_train_loss /= len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)
        
        # ==== 验证阶段 ====
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            val_loop = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} Validation')
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)
                val_loop.set_postfix(loss=loss.item())
        
        # 计算平均验证损失
        epoch_val_loss /= len(test_loader.dataset)
        val_loss_history.append(epoch_val_loss)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 更新学习率
        scheduler.step()
        
        # 打印epoch汇总信息
        print(f'  Epoch {epoch+1}/{num_epochs}:'
              f'  Train Loss: {epoch_train_loss:.6f}, '
              f'  Val Loss: {epoch_val_loss:.6f}, '
              f'  LR: {current_lr:.6f}')
        
        # 早停机制检查
        if epoch_val_loss < best_val_loss:
            print(f'    验证损失改进 ({best_val_loss:.6f} → {epoch_val_loss:.6f})')
            best_val_loss = epoch_val_loss
            best_model = model.state_dict().copy()  # 深拷贝模型权重
            counter = 0
            # 可选：保存当前epoch的模型
            torch.save(model.state_dict(), f'best_model_epoch_{epoch+1}.pth')
        else:
            counter += 1
            print(f'    验证损失未改善 ({counter}/{patience})')
            if counter >= patience:
                print(f'    验证损失连续{patience}个epoch未提升，停止训练')
                break
    
    # 保存最佳模型
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': best_model,
        'val_loss': best_val_loss,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history
    }, 'WCDMA_CNN_Best_Model.pth')
    print('\n最佳模型已保存为 WCDMA_CNN_Best_Model.pth')
    
    # 训练结束报告
    training_time = time.time() - training_start
    print(f'\n训练完成于第 {epoch+1}/{num_epochs} 个epoch，总耗时 {training_time/60:.2f} 分钟')
    
    # 绘制训练损失曲线
    plt.figure(figsize=(12, 8))
    
    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(train_loss_history, 'b-', linewidth=2, label='Training Loss')
    plt.plot(val_loss_history, 'r-', linewidth=2, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 学习率曲线
    plt.subplot(2, 1, 2)
    plt.plot(learning_rates, 'g-', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300)
    print('训练过程曲线已保存为 training_progress.png')
    
    # 绘制损失曲线 (仅损失)
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, 'b-', linewidth=2, label='Training Loss')
    plt.plot(val_loss_history, 'r-', linewidth=2, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
    print('损失曲线已保存为 loss_curve.png')
    
    # ==== BER评估 ====
    print("\n开始BER评估...")
    # 加载最佳模型
    model.load_state_dict(best_model)
    model.eval()  # 设置为评估模式
    
    # 按SNR计算BER
    ber_results = {}
    for snr in test_samples_by_snr:
        # 获取当前SNR的数据
        snr_samples = test_samples_by_snr[snr]
        snr_labels = test_labels_by_snr[snr]
        
        print(f'评估 SNR {snr} dB: {len(snr_samples)}个样本')
        print(f'  输入样本维度 (原始): {snr_samples.shape}')
        
        # 维度转换: (帧数, 高度, 宽度, 通道) -> (帧数, 通道, 高度, 宽度)
        adjusted_samples = np.transpose(snr_samples, (0, 3, 1, 2))
        print(f'  输入样本维度 (转换后): {adjusted_samples.shape}')
        
        # 转换为张量
        samples_tensor = torch.tensor(adjusted_samples, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(snr_labels, dtype=torch.float32).to(device)
        
        # 预测和计算BER
        with torch.no_grad():
            outputs = model(samples_tensor)
            print(f"  BER评估输出维度: {outputs.shape} (批次大小, 标签长度)")
            predictions = (outputs > 0.5).float().cpu().numpy()
            true_labels = labels_tensor.cpu().numpy()
            
            errors = np.sum(true_labels != predictions)
            total_bits = true_labels.size
            ber = errors / total_bits
            
            ber_results[snr] = ber
            print(f'  SNR {snr} dB - BER: {ber:.4f} (误码: {errors}/{total_bits})')
    
    # 绘制BER曲线
    snrs = sorted(ber_results.keys())
    bers = [ber_results[snr] for snr in snrs]
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(snrs, bers, 'o-', markersize=8, linewidth=2)
    plt.title('Bit Error Rate vs SNR', fontsize=14)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('ber_curve.png', dpi=300)
    print('BER曲线已保存为 ber_curve.png')
    
    total_time = time.time() - start_time
    print(f'\n总运行时间: {total_time/60:.2f} 分钟')

if __name__ == "__main__":
    main()