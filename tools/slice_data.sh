#!/bin/bash

# --- 配置参数 ---
# 待切片文件的路径
#INPUT_FILE="/home/zwang/parafilter-cuda/build/dataset/sift10m/train_vec"
INPUT_FILE="train_label_1"
# 输出文件的基础名称
OUTPUT_BASE_NAME="train_label_slice"
# Python 脚本路径
PYTHON_SCRIPT="./slice_data.py"
# 数据类型
DATA_TYPE="float32"
# 矩阵行数 N
N=9991000
# 矩阵列数 M
M=1
# 输出文件目录 (建议放在一个新目录，确保不会覆盖重要文件)
OUTPUT_DIR="./sliced_output"

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到 Python 脚本 $PYTHON_SCRIPT。请检查路径。"
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 找不到输入文件 $INPUT_FILE。请检查路径。"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo "输出文件将保存在: $OUTPUT_DIR"
echo "----------------------------------------"

# 循环 7 次，i 从 1 到 7
for i in {1..9}; do
    # 计算切分比例 ratio = i / 8
    # 使用 bc 进行浮点数计算，并保留4位小数
    RATIO=$(echo "scale=4; $i / 10" | bc)
    
    # 构造输出文件名: train_label_slice_0_125
    # 将小数点替换为下划线，用于文件名
    FILENAME_SUFFIX=$(echo "$RATIO" | sed 's/\./_/g')
    OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_BASE_NAME}_${FILENAME_SUFFIX}"
    
    echo "--- 第 $i 次切分: 比例 $i/10 = $RATIO ---"
    
    # 完整的 Python 命令
    COMMAND="python3 $PYTHON_SCRIPT $INPUT_FILE $OUTPUT_FILE $DATA_TYPE -N $N -M $M -r $RATIO"
    
    echo "正在执行: $COMMAND"
    
    # 执行命令
    # 注意: 假设 slice_data.py 是上一个回答中的 binary_slicer.py (按比例切片)
    $COMMAND
    
    # 检查上一个命令（Python 脚本）是否执行成功
    if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "警告: Python 脚本执行失败，停止循环。"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        break
    fi
    
    echo "" # 添加空行，分隔每次循环
done

echo "----------------------------------------"
echo "所有切分任务完成 (或因错误终止)。"