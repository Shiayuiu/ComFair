#!/bin/bash

# --- 配置区域 ---

# 数据集路径
RAW_DATA_PATH="./data/raw/adult.csv"
DATASET="adult"

# 模型参数
MODEL_TYPE="api"
MODEL_NAME="llama-3.1-8b-instruct"
MODEL_PATH="llama-3.1-8b-instruct"
BASE_URL= # 替换为实际的API URL
API_KEY="your_api_key_here" # 替换为你的API Key

# 任务参数
BATCH_SIZE=32
DEBUG_N=0 # 设置为 10 可以只跑前10条用于测试，设置为 0 跑全量

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
# skip

echo "========================================================"
echo "Step 2: Model Inference (Generation)"
echo "========================================================"
# 运行生成脚本
python run_gen.py \
    --dataset "$DATASET" \
    --model_type "$MODEL_TYPE" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --debug_n $DEBUG_N \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL"

if [ $? -eq 0 ]; then
    echo "========================================================"
    echo "Pipeline Finished Successfully!"
    echo "Check results in: data/results/predictions/"
    echo "========================================================"
else
    echo "Inference failed!"
    exit 1
fi