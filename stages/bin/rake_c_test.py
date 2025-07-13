import os
import h5py
import numpy as np
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

def main(data_path):
    """
    处理.h5文件计算不同SNR下的误码率
    
    参数:
    data_path: 主文件夹路径
    """
    # 存储结果
    snr_values = []
    ber_values = []
    
    # 遍历主文件夹
    folders = [f for f in os.listdir(data_path) 
              if os.path.isdir(os.path.join(data_path, f))]
    print(f"找到 {len(folders)} 个子文件夹")
    
    for folder in tqdm(folders, desc="处理SNR文件夹"):
        folder_path = os.path.join(data_path, folder)
        
        # 从文件夹名提取SNR值
        snr_match = re.search(r"snr_?([-\d.]+)", folder, re.IGNORECASE)
        if not snr_match:
            print(f"\n跳过不含SNR的文件夹: {folder}")
            continue
            
        snr = float(snr_match.group(1))
        total_errors = 0
        total_bits = 0
        processed_files = 0
        file_counter = 0
        
        # 获取当前SNR文件夹中的所有.h5文件
        files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
        print(f"\nSNR {snr}dB: 找到 {len(files)} 个.h5文件")
        
        # 处理当前SNR文件夹中的所有.h5文件
        for file in tqdm(files, desc=f"SNR {snr}dB 文件处理"):
            file_path = os.path.join(folder_path, file)
            try:
                with h5py.File(file_path, 'r') as f:
                    # 读取所需字段
                    fields = ['finger_data_channel_signal', 'channel_estimates', 'original_bit']
                    data = {}
                    missing_fields = []
                    
                    # 检查并读取所有字段
                    for field in fields:
                        if field in f:
                            data[field] = f[field][:]
                        else:
                            missing_fields.append(field)
                    
                    # 如果有缺失字段则跳过该文件
                    if missing_fields:
                        if file_counter < 3:  # 仅打印前3个文件的错误
                            print(f"\n文件 {file} 缺少字段: {', '.join(missing_fields)}")
                        file_counter += 1
                        continue
                    
                    # 打印前几个文件的数据维度用于调试
                    if processed_files < 3:
                        print(f"\n文件 {file} 数据维度:")
                        for field, array in data.items():
                            print(f"  - {field}: {array.shape} (dtype: {array.dtype})")
                    
                    # 提取数据
                    finger_signal = data['finger_data_channel_signal']  # 多径信号
                    channel_est = data['channel_estimates']  # 信道估计
                    original_bits = data['original_bit']  # 原始比特
                    
                    # 检查维度是否匹配
                    if finger_signal.shape != channel_est.shape:
                        print(f"\n错误: 文件 {file} 信号和信道估计维度不匹配")
                        print(f"  信号形状: {finger_signal.shape}, 信道估计形状: {channel_est.shape}")
                        continue
                    
                    # 使用MRC合并多径信号
                    # 共轭信道估计乘以信号
                    mrc_signal = np.conj(channel_est) * finger_signal
                    
                    # 合并多径: 求和 (最大比合并)
                    combined_signal = np.sum(mrc_signal, axis=0)
                    
                    # 硬判决 (假设BPSK调制)
                    # 只取实部，因为虚部可能是噪声
                    hard_decision = np.where(np.real(combined_signal) >= 0, 1, 0)
                    
                    # 检查比特长度是否匹配
                    if len(hard_decision) != len(original_bits):
                        print(f"\n错误: 文件 {file} 判决后比特与原始比特长度不匹配")
                        print(f"  判决比特长度: {len(hard_decision)}, 原始比特长度: {len(original_bits)}")
                        continue
                    
                    # 计算误码
                    errors = np.sum(hard_decision != original_bits)
                    
                    # 更新统计
                    total_errors += errors
                    total_bits += len(original_bits)
                    processed_files += 1
                    file_counter += 1
                    
            except Exception as e:
                print(f"\n处理 {file} 时出错: {str(e)}")
        
        # 计算当前SNR的BER
        if processed_files == 0:
            print(f"\nSNR {snr}dB: 无有效文件")
            continue
            
        if total_bits == 0:
            print(f"\nSNR {snr}dB: 总比特数为零")
            continue
            
        ber = total_errors / total_bits
        snr_values.append(snr)
        ber_values.append(ber)
        print(f"\nSNR: {snr}dB | BER: {ber:.3e} | 处理文件数: {processed_files}/{len(files)} | 总比特数: {total_bits} | 总误码: {total_errors}")
    
    if not snr_values:
        print("\n错误: 未处理任何SNR数据")
        return
    
    # 按SNR排序结果
    sorted_indices = np.argsort(snr_values)
    snr_values = np.array(snr_values)[sorted_indices]
    ber_values = np.array(ber_values)[sorted_indices]
    
    # 输出最终结果
    print("\n" + "="*50)
    print("最终结果:")
    print("="*50)
    for snr, ber in zip(snr_values, ber_values):
        print(f"SNR {snr:.1f}dB: BER = {ber:.5e}")
    
    # 绘制BER曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_values, ber_values, 'o-', linewidth=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs SNR')
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    
    # 保存结果
    output_file = os.path.join(data_path, 'ber_results.png')
    plt.savefig(output_file)
    print(f"\nBER曲线已保存至: {output_file}")
    
    # 同时保存文本结果
    result_file = os.path.join(data_path, 'ber_results.txt')
    with open(result_file, 'w') as f:
        f.write("SNR(dB)\tBER\n")
        for snr, ber in zip(snr_values, ber_values):
            f.write(f"{snr:.1f}\t{ber:.6e}\n")
    print(f"文本结果已保存至: {result_file}")
    
    plt.show()
    
    return snr_values, ber_values

# 使用示例
if __name__ == "__main__":
    data_path = "/data/duhu/WCDM/data_stages/rician_channel/fraction_delay"  # 替换为你的主文件夹路径
    print("="*50)
    print("开始处理误码率计算")
    print("="*50)
    snr, ber = main(data_path)