#!/bin/bash

# --- 配置参数 ---

# 1. 新增：源文件所在的目录
SOURCE_BASE_DIR="./sliced_output"

# 源文件名的基础前缀（不带目录）
SOURCE_BASE_LABEL="train_label_slice__"
SOURCE_BASE_VEC="train_vec_slice__"

# 目标文件的基础目录 (不含最后的数字 /1, /2, ...)
TARGET_BASE_DIR="/home/zwang/parafilter-cuda/build/dataset/sift10m_1label/"

# 比例因子：i/8，用于计算文件后缀
# 1/8 = 0.1250 -> 1250
# 7/8 = 0.8750 -> 8750

echo "--- 开始移动文件 ---"
echo "源文件目录: $SOURCE_BASE_DIR"
echo "目标根目录: $TARGET_BASE_DIR"
echo "----------------------"

# 检查源目录是否存在
if [ ! -d "$SOURCE_BASE_DIR" ]; then
    echo "错误: 源目录 '$SOURCE_BASE_DIR' 不存在。请确保你在这个目录下运行脚本。"
    exit 1
fi

# 循环 7 次，i 从 1 到 7
for i in {1..9}; do
    # 1. 计算文件后缀 (1250, 2500, 3750, ..., 8750)
    SUFFIX=$((i * 1000))

    # 2. 构造完整的源文件路径 (重点修改处)
    # 例如: ./sliced_output/train_label_slice_2500
    SOURCE_LABEL="${SOURCE_BASE_DIR}/${SOURCE_BASE_LABEL}${SUFFIX}"
    SOURCE_VEC="${SOURCE_BASE_DIR}/${SOURCE_BASE_VEC}${SUFFIX}"
    
    # 3. 构造目标目录 (例如: /.../sift10m/slice/2)
    TARGET_DIR="${TARGET_BASE_DIR}/${i}"

    # 4. 创建目标目录及其所有父目录
    mkdir -p "$TARGET_DIR"

    echo "处理 [$i/10]: 目标目录 ${TARGET_DIR}"

    # --- 移动第一个文件: train_label ---
    if [ -f "$SOURCE_LABEL" ]; then
        echo "  -> 移动 ${SOURCE_LABEL} 到 ${TARGET_DIR}/train_label"
        mv "$SOURCE_LABEL" "${TARGET_DIR}/train_label"
    else
        echo "  警告: 源文件 ${SOURCE_LABEL} 不存在，跳过。"
    fi
    
    # --- 移动第二个文件: train_vec ---
    if [ -f "$SOURCE_VEC" ]; then
        echo "  -> 移动 ${SOURCE_VEC} 到 ${TARGET_DIR}/train_vec"
        mv "$SOURCE_VEC" "${TARGET_DIR}/train_vec"
    else
        echo "  警告: 源文件 ${SOURCE_VEC} 不存在，跳过。"
    fi
    
    echo "" # 添加空行分隔
done

echo "--- 所有移动操作完成 ---"