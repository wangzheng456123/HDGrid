import numpy as np
import os
import sys

# ----------------------------------------------------------------------
# 配置参数
# ----------------------------------------------------------------------
base_path = "/home/zwang/parafilter-cuda/build/dataset/sift10m/slice/5/"

# 输入文件列表
INPUT_FILES = ["train_label", "train_vec", "test_label", "test_vec"]

# 对应的输出文件列表 (fvecs 格式)
OUTPUT_FILES = ["sift5m_labels.fvecs", "sift5m_base.fvecs", "sift5m_filters.fvecs", "sift5m_query.fvecs"]

# 定义数据类型和维度 (对于标签/ID文件，M=1)
# 注意：你需要确保你的二进制文件的实际类型是正确的
FILE_CONFIGS = {
    # 标签文件：通常是 int32 或 uint32，fvecs 要求向量是 float32，但标签/ID文件通常 M=1
    "train_label": {"dtype": np.float32, "M": 3},
    "test_label":  {"dtype": np.float32, "M": 6},
    
    # 向量文件：通常是 float32，维度 M=128
    "train_vec":   {"dtype": np.float32, "M": 128},
    "test_vec":    {"dtype": np.float32, "M": 128},
}

# ----------------------------------------------------------------------
# 核心函数 (与之前的一致)
# ----------------------------------------------------------------------

def bin_to_fvecs(input_bin_path: str, output_fvecs_path: str, N: int, M: int, dtype: np.dtype) -> bool:
    """
    将 N * M 的矩阵二进制文件转换为 fvecs 格式文件。
    """
    print(f"\n--- 正在转换文件: {os.path.basename(input_bin_path)} ---")
    
    # 1. 动态计算文件大小和预期的 N
    itemsize = dtype().itemsize
    expected_bytes = N * M * itemsize
    
    try:
        file_size = os.path.getsize(input_bin_path)
    except FileNotFoundError:
        print(f"错误: 找不到输入文件: {input_bin_path}", file=sys.stderr)
        return False

    if file_size != expected_bytes:
        print(f"警告: 文件大小 {file_size} 字节与预期大小 {expected_bytes} 字节不匹配。", file=sys.stderr)
        # 重新计算 N 以匹配实际文件大小
        actual_N = file_size // (M * itemsize)
        print(f"将根据实际文件大小读取 {actual_N} 个向量 (而非预期的 {N} 个)。", file=sys.stderr)
        N = actual_N

    if N == 0:
        print("文件为空，跳过转换。", file=sys.stderr)
        return True # 视为成功

    # 2. 读取数据
    try:
        data = np.fromfile(input_bin_path, dtype=dtype)
        matrix = data.reshape(N, M)
    except Exception as e:
        print(f"错误: 读取或重塑文件时发生错误: {e}", file=sys.stderr)
        return False

    print(f"成功读取 {N} 个 {M} 维向量。")

    # 3. 写入 fvecs 格式
    
    # M 的 32 位整数二进制表示
    dim_M_int32 = np.array([M], dtype=np.int32) 
    dim_M_bytes = dim_M_int32.tobytes() 

    # 目标 fvecs 格式要求数据是 float32。如果原始数据不是，需要先转换。
    matrix_float32 = matrix.astype(np.float32)

    try:
        with open(output_fvecs_path, 'wb') as f:
            for i in range(N):
                vector = matrix_float32[i, :]
                
                # 写入：[int32: M] (维度)
                f.write(dim_M_bytes)
                
                # 写入：[float32 * M] (向量数据)
                f.write(vector.tobytes())
                
    except Exception as e:
        print(f"错误: 写入文件 {output_fvecs_path} 时发生错误: {e}", file=sys.stderr)
        return False
        
    print(f"转换完成！已写入 {N} 个向量到 {output_fvecs_path}")
    return True

# ----------------------------------------------------------------------
# 验证函数 (与之前的一致)
# ----------------------------------------------------------------------

def read_fvecs_to_numpy(filename):
    """读取 fvecs 文件并返回 NumPy 数组。"""
    a = np.fromfile(filename, dtype='int32')
    if a.size == 0:
        return np.zeros((0, 0))
    dim = a[0]
    matrix = a.reshape(-1, dim + 1)
    if not np.all(matrix[:, 0] == dim):
        raise IOError("fvecs 文件中向量维度不一致！")
    return matrix[:, 1:].copy().view('float32')

# ----------------------------------------------------------------------
# 主执行逻辑 (列表遍历)
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    # 假设所有文件都有相同数量的 N_VECTORS (如果不同，需要单独确定)
    # 警告：这里使用一个固定的 N 值，如果你的不同文件 N 值不同，你需要调整这个逻辑。
    # 如果 train_label 和 train_vec 行数相同，但 test_* 行数不同，你需要单独计算。
    # 由于原始代码提供了 N_VECTORS = 1000 作为示例，我们暂时保留，但更推荐动态计算 N。
    #
    # *** 最佳实践：如果 N 是未知且不同的，应注释掉 N_VECTORS，并在 bin_to_fvecs 内部
    # *** 通过文件大小 / (M * itemsize) 来动态计算 N。
    N_VECTORS_DEFAULT = 1000 
    
    all_success = True
    
    # 使用 zip 遍历两个列表
    for input_name, output_name in zip(INPUT_FILES, OUTPUT_FILES):
        
        # 组合完整路径
        input_path = os.path.join(base_path, input_name)
        output_path = os.path.join(base_path, output_name)
        
        # 获取该文件的配置
        if input_name in FILE_CONFIGS:
            config = FILE_CONFIGS[input_name]
            dtype = config["dtype"]
            M_DIMENSION = config["M"]
        else:
            # 如果配置中没有，跳过或使用默认值
            print(f"错误: 缺少文件 {input_name} 的配置 (dtype 和 M)。跳过。", file=sys.stderr)
            all_success = False
            continue

        # 确定 N 值
        # 对于标签/向量对，我们通常假设它们有相同的 N。
        # 如果是 train/test 对，N_VECTORS_DEFAULT 不应该用于所有文件。
        # 为了通用性，我们在这里**假设 N 未知**，并让 bin_to_fvecs 内部计算 N。
        # 如果你想使用固定的 N_VECTORS_DEFAULT，请取消注释下一行，并调整 bin_to_fvecs 逻辑。
        # N = N_VECTORS_DEFAULT
        N = N_VECTORS_DEFAULT # 让函数内部通过文件大小计算 N

        # 执行转换
        if not bin_to_fvecs(input_path, output_path, N, M_DIMENSION, dtype):
            all_success = False
            
        # 验证输出文件
        print("\n--- 验证输出文件 ---")
        try:
            loaded_matrix = read_fvecs_to_numpy(output_path)
            print(f"成功从 {os.path.basename(output_path)} 读取矩阵。")
            print(f"形状: {loaded_matrix.shape}")
            print(f"数据类型: {loaded_matrix.dtype}")
        except Exception as e:
            print(f"错误: 验证 fvecs 文件 {os.path.basename(output_path)} 时发生错误: {e}", file=sys.stderr)
            all_success = False
            
    if all_success:
        print("\n所有文件转换和验证均已完成！")
    else:
        print("\n警告: 部分文件转换失败。", file=sys.stderr)