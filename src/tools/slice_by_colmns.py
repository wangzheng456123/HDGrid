import argparse
import numpy as np
import os
import sys
from typing import Type

def slice_matrix_columns(
    input_filename: str,
    output_filename: str,
    dtype: Type,
    N: int,
    M: int,
    K: int
) -> None:
    """
    读取 N x M 矩阵的二进制文件，取出前 K 列，并存储到新文件。
    该函数使用 NumPy 的内存映射 (memmap) 来高效处理大文件。
    
    Args:
        input_filename: 要读取的二进制文件路径。
        output_filename: 要写入的新二进制文件路径。
        dtype: 文件中存储的数据类型 (例如 np.float64, np.int32)。
        N: 矩阵的行数。
        M: 矩阵的列数。
        K: 要取出的前 K 列的数量。
    
    Raises:
        FileNotFoundError: 如果输入文件不存在。
        ValueError: 如果 K 大于 M。
        RuntimeError: 如果文件大小与预期不匹配。
    """
    
    if K > M:
        raise ValueError(f"要取出的列数 K ({K}) 不能大于矩阵的总列数 M ({M})。")

    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"错误: 找不到输入文件 '{input_filename}'。")

    # --- 1. 验证文件大小 ---
    total_elements = N * M
    itemsize = np.dtype(dtype).itemsize
    
    expected_size = total_elements * itemsize
    actual_size = os.path.getsize(input_filename)
    
    if actual_size != expected_size:
        raise RuntimeError(
            f"文件大小不匹配。预期 {expected_size} 字节 ({N}x{M}*{np.dtype(dtype).name}), "
            f"实际 {actual_size} 字节。"
        )
    
    print(f"--- 切片信息 ---")
    print(f"矩阵维度: {N} 行 x {M} 列")
    print(f"数据类型: {np.dtype(dtype).name} ({itemsize} 字节)")
    print(f"目标切片: 前 {K} 列 ({N} 行 x {K} 列)")
    print(f"------------------")

    # --- 2. 使用 NumPy Mmap 读取、切片和写入 ---
    try:
        # 使用 numpy.memmap 以内存映射方式打开文件
        # 'r' 表示只读，将文件视为 N x M 的二维数组
        # 'C' order (行主序) 是默认的，适用于大多数C/C++生成的二进制文件
        data_mmap = np.memmap(
            input_filename, 
            dtype=dtype, 
            mode='r', 
            shape=(N, M), 
            order='C'
        )
        
        # 执行列切片操作：取出所有行 (:) 的前 K 列 (:K)
        # 注意：这个操作会创建一个数据的*副本*，因为列是不连续的
        sliced_data = data_mmap[:, :K]
        
        # 将切片后的副本数据直接写入新文件
        # 'w+b' 表示以二进制方式写入，如果文件存在则覆盖
        sliced_data.tofile(output_filename)
        
        # 释放内存映射
        del data_mmap
        
    except Exception as e:
        raise RuntimeError(f"处理文件时发生错误: {e}")

    print(f"\n✅ 成功将 {N}x{K} 的矩阵写入 '{output_filename}'。")


def main():
    """主函数，处理命令行参数。"""
    parser = argparse.ArgumentParser(
        description="一个高效的 N x M 二进制矩阵列切片工具。",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('input_file', type=str, help="待切片的输入二进制文件路径。")
    parser.add_argument('output_file', type=str, help="切片后输出的新文件路径。")
    parser.add_argument('data_type', type=str, 
                        help="文件中存储的数据类型。\n"
                             "常用类型: float64, float32, int64, int32, etc.")
    parser.add_argument('-N', type=int, required=True, 
                        help="矩阵的行数 (N)。")
    parser.add_argument('-M', type=int, required=True, 
                        help="矩阵的总列数 (M)。")
    parser.add_argument('-K', type=int, required=True, 
                        help="要取出的前 K 列的数量。")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    try:
        # 将字符串类型转换为 NumPy 的数据类型对象
        np_dtype = np.dtype(args.data_type)
        
        slice_matrix_columns(
            input_filename=args.input_file,
            output_filename=args.output_file,
            dtype=np_dtype,
            N=args.N,
            M=args.M,
            K=args.K
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)
    except TypeError:
        print(f"\n❌ 错误: 无法识别数据类型 '{args.data_type}'。请检查类型名称。", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    # ---------------------- 示例：创建测试文件 ----------------------
    try:
        input_test_file = "matrix_input.bin"
        N_test, M_test = 5, 8
        
        if not os.path.exists(input_test_file):
            # 创建一个 5x8 的矩阵 (40 个元素)
            # 元素值为 0, 1, 2, ..., 39
            test_data = np.arange(N_test * M_test, dtype=np.int32).reshape(N_test, M_test)
            test_data.tofile(input_test_file)
            print(f"已创建测试文件 '{input_test_file}' ({N_test}x{M_test} int32 矩阵)。")
            print("原始矩阵 (部分):\n", test_data)
    except Exception as e:
        print(f"创建测试文件失败: {e}", file=sys.stderr)
        
    main()