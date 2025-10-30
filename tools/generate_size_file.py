import struct
import os

# 32位整数占用 4 字节
INT_SIZE = 4
# 每个文件固定包含 2 个整数
NUM_INTS_PER_FILE = 2
FILE_SIZE_BYTES = NUM_INTS_PER_FILE * INT_SIZE # 固定为 8 字节

def generate_two_int_binary_file(filename, int_value_1, int_value_2):
    """
    生成一个固定大小为 8 字节的二进制文件，包含两个 32 位整数。

    :param filename: 输出文件名。
    :param int_value_1: 写入的第一个整数值。
    :param int_value_2: 写入的第二个整数值。
    """
    
    # 使用小端字节序 '<' 和两个 32位整数 'ii'
    # 'ii' 表示两个连续的 signed int
    # 如果需要无符号整数，请使用 'II'
    packer = struct.Struct('<ii')
    
    # 打包要写入的二进制数据块
    data_chunk = packer.pack(int_value_1, int_value_2)

    print(f"-> 正在创建文件: {filename}")
    print(f"   固定大小: {FILE_SIZE_BYTES} 字节 (包含两个 int32: {int_value_1}, {int_value_2})")

    try:
        with open(filename, 'wb') as f:
            f.write(data_chunk)
            
        # 验证文件大小是否正确
        if os.path.getsize(filename) != FILE_SIZE_BYTES:
            print(f"   警告: 文件大小验证失败，预期 {FILE_SIZE_BYTES} 字节，实际 {os.path.getsize(filename)} 字节。")
        else:
            print(f"   成功生成文件，实际大小: {os.path.getsize(filename)} 字节。")

    except Exception as e:
        print(f"   错误: 写入文件失败: {e}")

def main():
    # --- 配置区域 ---
    
    # 1. 输出目录
    OUTPUT_DIR = "generated_two_int_data"
    
    # 2. 要生成的二进制文件列表
    # 格式: "文件名": [第一个 int 值, 第二个 int 值]
    files_to_generate = {
        "test_label_size": [10000, 1],   # 包含整数 -1 和 999999
    }
    
    # --- 脚本逻辑 ---
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"将文件生成到目录: {OUTPUT_DIR}")
    print(f"每个文件固定大小为 {FILE_SIZE_BYTES} 字节。")
    print("------------------------------------------")

    for filename, values in files_to_generate.items():
        if len(values) != NUM_INTS_PER_FILE:
            print(f"跳过文件 {filename}: 需要 {NUM_INTS_PER_FILE} 个值，但提供了 {len(values)} 个。")
            continue
            
        full_path = os.path.join(OUTPUT_DIR, filename)
        generate_two_int_binary_file(full_path, values[0], values[1])
        print("------------------------------------------")

if __name__ == "__main__":
    main()