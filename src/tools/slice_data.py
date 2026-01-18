import argparse
import numpy as np
import os
import sys
from typing import Type

def slice_binary_file(
    input_filename: str,
    output_filename: str,
    dtype: Type,
    N: int,
    M: int,
    ratio: float
) -> None:
    """
    读取二进制文件，按指定比例切片数据，并写入新文件。
    
    Args:
        input_filename: 要读取的二进制文件路径。
        output_filename: 要写入的新二进制文件路径。
        dtype: 文件中存储的数据类型 (例如 np.float64, np.int32)。
        N: 数据的行数（块数）。
        M: 每行/块的元素数量。
        ratio: 要保留的数据比例 (0.0 到 1.0)。
    
    Raises:
        FileNotFoundError: 如果输入文件不存在。
        ValueError: 如果 ratio 不在 [0.0, 1.0] 范围内。
        RuntimeError: 如果文件大小与预期不匹配。
    """
    
    if not (0.0 <= ratio <= 1.0):
        raise ValueError("切片比例 ratio 必须在 0.0 到 1.0 之间。")

    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"错误: 找不到输入文件 '{input_filename}'。")

    # --- 1. 计算切片大小 ---
    total_elements = N * M
    # 计算需要保留的元素数量，并取整
    slice_elements = int(np.round(total_elements * ratio))
    
    # 获取数据类型的大小
    itemsize = np.dtype(dtype).itemsize
    
    # 检查文件大小是否匹配
    expected_size = total_elements * itemsize
    actual_size = os.path.getsize(input_filename)
    
    if actual_size != expected_size:
        print(f"警告: 文件 '{input_filename}' 实际大小为 {actual_size} 字节，"
              f"与预期大小 {expected_size} 字节 (N*M*sizeof(dtype)) 不匹配。将尝试读取。")
    
    print(f"--- 切片信息 ---")
    print(f"总元素数量: {total_elements}")
    print(f"数据类型: {np.dtype(dtype).name} ({itemsize} 字节)")
    print(f"保留比例: {ratio * 100:.2f}%")
    print(f"目标切片元素数量: {slice_elements}")
    print(f"目标切片大小: {slice_elements * itemsize / (1024*1024):.2f} MB")
    print(f"------------------")

    # --- 2. 使用 NumPy Mmap 高效读取/切片/写入 ---
    try:
        # 使用 numpy.memmap 以内存映射方式打开文件，高效处理大文件
        # 'r' 表示只读
        # shape=(total_elements,) 将文件视为一维数组
        data = np.memmap(input_filename, dtype=dtype, mode='r', shape=(total_elements,))
        
        # 执行切片操作
        # data[:slice_elements] 实现了取出前 N * ratio * M 大小的切片
        sliced_data = data[:slice_elements]
        
        # 将切片后的数据直接写入新文件
        # 'w+b' 表示以二进制方式写入，如果文件存在则覆盖
        # 注: 如果 slice_elements 为 0，sliced_data 为空，文件会被创建但内容为空
        sliced_data.tofile(output_filename)
        
        # 释放内存映射
        del data
        
    except Exception as e:
        raise RuntimeError(f"处理文件时发生错误: {e}")

    print(f"\n✅ 成功将 {slice_elements} 个元素 (前 {ratio*100:.2f}%) 写入 '{output_filename}'。")


def main():
    """主函数，处理命令行参数。"""
    parser = argparse.ArgumentParser(
        description="一个高效的二进制文件切片工具，使用 NumPy Mmap。",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('input_file', type=str, help="待切片的输入二进制文件路径。")
    parser.add_argument('output_file', type=str, help="切片后输出的新文件路径。")
    parser.add_argument('data_type', type=str, 
                        help="文件中存储的数据类型。\n"
                             "常用类型: float64, float32, int64, int32, uint64, uint32, etc.")
    parser.add_argument('-N', type=int, required=True, 
                        help="数据块/行数。")
    parser.add_argument('-M', type=int, required=True, 
                        help="每块/行中的元素数量。")
    parser.add_argument('-r', '--ratio', type=float, default=0.1, 
                        help="要保留的数据比例 (0.0 到 1.0)。默认为 0.1 (10%%)。")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    try:
        # 将字符串类型转换为 NumPy 的数据类型对象
        np_dtype = np.dtype(args.data_type)
        
        slice_binary_file(
            input_filename=args.input_file,
            output_filename=args.output_file,
            dtype=np_dtype,
            N=args.N,
            M=args.M,
            ratio=args.ratio
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)
    except TypeError:
        print(f"\n❌ 错误: 无法识别数据类型 '{args.data_type}'。请检查类型名称。", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    # 示例：创建测试文件
    try:
        if not os.path.exists("test_input.bin"):
            N_test, M_test = 10, 10
            total_test = N_test * M_test
            test_data = np.arange(total_test, dtype=np.float64)
            test_data.tofile("test_input.bin")
            print(f"已创建测试文件 'test_input.bin' (100个 float64 元素)。")
    except Exception as e:
        print(f"创建测试文件失败: {e}", file=sys.stderr)
        
    main()