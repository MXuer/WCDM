import h5py
import numpy as np
import os

# 文件路径
input_file_path = "/data/duhu/WCDM/raw_data/Uniform_SF16_unrelated_fraction_Train_dataSet_160Bit_HDF5_20250621_094320/mutiPath_train_data.h5"
output_file_path = "/data/duhu/WCDM/raw_data/Uniform_SF16_unrelated_fraction_Train_dataSet_160Bit_HDF5_20250621_094320/frame_verify_data.h5"
screamble_file_path = "/data/duhu/WCDM/raw_data/spreading_screamble_uprate_result.h5"


# 设置输出目录
output_directory = "/data/duhu/WCDM/data_new/onepath_spilt_dataset"
# 创建输出目录（如果不存在）
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"已创建输出目录: {output_directory}")


def read_and_process_h5(file_path, is_input_data=False):
    """读取HDF5文件，转置数据集，并对输入数据进行第四维度切片"""
    try:
        print(f"\n处理文件: {file_path}")
        print("=" * 50)
        
        with h5py.File(file_path, 'r') as f:
            # 递归处理所有数据集
            results = {}
            
            for name in f:
                if isinstance(f[name], h5py.Dataset):
                    print(f"数据集: {name}")
                    print(f"  原始维度: {f[name].shape}")
                    
                    # 读取数据并进行转置
                    data = f[name][()]
                    transposed_data = data.T.copy()
                    
                    print(f"  转置后维度: {transposed_data.shape}")
                    
                    # 对输入数据进行额外处理：取第四维度的第一个元素
                    if is_input_data and transposed_data.ndim == 4:
                        print("  处理输入数据：取第四维度(索引=0)")
                        processed_data = transposed_data[..., 0]  # 等同于 transposed_data[:, :, :, 0]
                        print(f"  处理后维度: {processed_data.shape}")
                        results[name] = processed_data
                    else:
                        results[name] = transposed_data
                    
            return results
            
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return None

def insert_screamble_data(input_data, screamble_data):
    """在输入数据的第二维插入扰码数据"""
    print("\n开始插入扰码数据...")
    
    # 获取输入数据
    input_name = list(input_data.keys())[0]
    input_array = input_data[input_name]
    
    # 获取扰码数据
    screamble_name = list(screamble_data.keys())[0]
    screamble_array = screamble_data[screamble_name]
    
    print(f"输入数据形状: {input_array.shape}")
    print(f"扰码数据形状: {screamble_array.shape}")
    
    # 确保扰码数据是2×10240
    if screamble_array.shape != (2, 10240):
        # 尝试转置或重塑
        if screamble_array.shape == (10240, 2):
            screamble_array = screamble_array.T
            print(f"扰码数据已转置为: {screamble_array.shape}")
        else:
            raise ValueError(f"扰码数据形状{screamble_array.shape}不符合要求(2, 10240)")
    
    # 复制扰码数据以匹配样本数量
    num_samples = input_array.shape[0]
    expanded_screamble = np.tile(screamble_array, (num_samples, 1, 1))
    print(f"扩展后的扰码数据形状: {expanded_screamble.shape}")
    
    # 在第二维拼接输入数据和扰码数据
    combined_data = np.concatenate((input_array, expanded_screamble), axis=1)
    print(f"拼接后数据形状: {combined_data.shape}")
    
    return {input_name: combined_data}


def reshape_input_output(combined_input, output_data):
    """处理组合后的输入数据和输出数据"""
    print("\n开始处理输入输出数据...")
    
    # 获取输入数据
    input_name = list(combined_input.keys())[0]
    input_array = combined_input[input_name]
    num_samples, dim1, dim2 = input_array.shape
    print(f"输入数据形状: {input_array.shape}")
    
    # 获取输出数据
    output_name = list(output_data.keys())[0]
    output_array = output_data[output_name]
    print(f"输出数据形状: {output_array.shape}")
    
    # 检查数据一致性
    if num_samples != output_array.shape[0]:
        raise ValueError(f"样本数量不一致: 输入有{num_samples}个样本, 输出有{output_array.shape[0]}个样本")
    
    # 计算每份切割的数量
    chunk_size = 64
    chunks_per_sample = dim2 // chunk_size
    
    # 初始化结果数组
    processed_input = np.zeros((num_samples * chunks_per_sample, dim1, chunk_size), dtype=input_array.dtype)
    processed_output = np.zeros((num_samples * chunks_per_sample, 1), dtype=output_array.dtype)
    
    # 处理每个样本
    for sample_idx in range(num_samples):
        # 处理输入数据
        sample_input = input_array[sample_idx]  # (6, 10240)
        
        # 切分为chunks_per_sample份 (每份6, 64)
        chunks = [sample_input[:, i*chunk_size:(i+1)*chunk_size] for i in range(chunks_per_sample)]
        
        # 重组为(160, 6, 64) 然后按顺序存储
        for chunk_idx, chunk in enumerate(chunks):
            output_idx = sample_idx * chunks_per_sample + chunk_idx
            processed_input[output_idx] = chunk
        
        # 处理输出数据
        sample_output = output_array[sample_idx]  # (160,)
        
        # 转换为(160, 1)并按顺序存储
        for chunk_idx in range(chunks_per_sample):
            output_idx = sample_idx * chunks_per_sample + chunk_idx
            processed_output[output_idx, 0] = sample_output[chunk_idx]
        
        # 每处理100个样本打印一次进度
        if (sample_idx + 1) % 100 == 0 or sample_idx == num_samples - 1:
            print(f"已处理 {sample_idx+1}/{num_samples} 个样本")
    
    print(f"处理后的输入数据形状: {processed_input.shape}")
    print(f"处理后的输出数据形状: {processed_output.shape}")
    
    return processed_input, processed_output

if __name__ == "__main__":
    print("读取输入数据...")
    input_data = read_and_process_h5(input_file_path, is_input_data=True)
    
    print("\n读取输出数据...")
    output_data = read_and_process_h5(output_file_path, is_input_data=False)
    
    print("\n读取扰码数据...")
    screamble_data = read_and_process_h5(screamble_file_path, is_input_data=False)
    
if input_data and output_data and screamble_data:
        print("\n数据处理完成!")
        
        # 插入扰码数据
        combined_input = insert_screamble_data(input_data, screamble_data)
        
        # 获取组合后的输入数据
        input_name = list(combined_input.keys())[0]
        input_array = combined_input[input_name]
        
        # 处理输入输出数据
        processed_input, processed_output = reshape_input_output(combined_input, output_data)
        
        # 生成输出文件路径（在指定的目录中）
        processed_file_path = os.path.join(output_directory, "train.h5")

        # 保存处理后的数据
        with h5py.File(processed_file_path, 'w') as f_out:
            f_out.create_dataset("input", data=processed_input)
            f_out.create_dataset("output", data=processed_output)
        
        print(f"已保存最终处理数据到: {processed_file_path}")
        
        # # 打印一些示例数据
        # print("\n数据示例验证:")
        # print(f"第一个输入块 (0:6, 0:64):\n{processed_input[0]}")
        # print(f"对应的输出值: {processed_output[0][0]}")
        
        # print(f"\n第160个输入块 (160:166, 0:64):\n{processed_input[160]}")
        # print(f"对应的输出值: {processed_output[160][0]}")
        
        # print(f"\n第161个输入块 (161:167, 0:64):\n{processed_input[161]}")
        # print(f"对应的输出值: {processed_output[161][0]}")