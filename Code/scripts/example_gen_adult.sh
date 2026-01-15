#!/bin/bash

# --- 配置区域 ---
# 设置显卡 (可选)
export CUDA_VISIBLE_DEVICES=1

# 数据集路径
RAW_DATA_PATH="./data/raw/adult.csv"

# 模型路径 (可以是 HuggingFace ID，也可以是绝对路径)

# MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct" 
# your model path
MODEL_PATH="/zhdd/home/yfwu/model/Qwen2.5-1.5B-Instruct/"

# 任务参数
BATCH_SIZE=8
DEBUG_N=0  # 设置为 10 可以只跑前10条用于测试，设置为 0 跑全量

# --- 执行区域 ---

echo "========================================================"
echo "Step 0: Checking Environment"
echo "========================================================"
# 简单的目录检查
if [ ! -f "$RAW_DATA_PATH" ]; then
    echo "Error: Raw data not found at $RAW_DATA_PATH"
    exit 1
fi

echo "========================================================"
echo "Step 1: Data Preprocessing"
echo "========================================================"
# 运行数据准备脚本

echo "========================================================"
echo "Step 2: Model Inference (Generation)"
echo "========================================================"
# 运行生成脚本
python run_gen.py \
    --dataset adult \
    --model_name "Qwen2.5-1.5B" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --debug_n $DEBUG_N

if [ $? -eq 0 ]; then
    echo "========================================================"
    echo "Pipeline Finished Successfully!"
    echo "Check results in: data/results/predictions/"
    echo "========================================================"
else
    echo "Inference failed!"
    exit 1
fi