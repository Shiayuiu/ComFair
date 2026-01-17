#!/bin/bash

# --- 配置区域 ---
# 设置显卡 (可选)
export CUDA_VISIBLE_DEVICES=1

# Dataset settings
RAW_DATA_PATH="./data/raw/adult.csv"
DATASET="adult"

# Model settings
MODEL_NAME="Qwen2.5-1.5B"
MODEL_PATH="/zhdd/home/yfwu/model/Qwen2.5-1.5B-Instruct/"

# 任务参数
BATCH_SIZE=8
DEBUG_N=0  # 设置为N:运行前N条用于测试;设置为0:跑全量

# 查询参数 (Query Stage)
# 1. 如果留空 ""，则脚本会调用 configs/[dataset].py 中默认定义的敏感属性
# 这里留空则指向 configs/adult.py 中的敏感属性列表
# 2. 或者手动指定: QUERY_ATTRS="race gender age_group"
QUERY_ATTRS="race gender"

# 结果路径配置
PRED_DIR="./data/results/predictions/"

# --- 执行区域 ---

echo "========================================================"
echo "Gen Step 0: Checking Environment"
echo "========================================================"
# 目录检查
if [ ! -f "$RAW_DATA_PATH" ]; then
    echo "Error: Raw data not found at $RAW_DATA_PATH"
    exit 1
fi

echo "========================================================"
echo "Gen Step 1: Data Preprocessing"
echo "========================================================"
# 运行数据准备脚本

echo "========================================================"
echo "Gen Step 2: Model Inference (Generation)"
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
    echo "Gen Finished Successfully!"
    echo "Check results in: data/results/predictions/"
    echo "========================================================"
else
    echo "Inference failed!"
    exit 1
fi



# --- 执行区域 ---

echo "========================================================"
echo "Eval Step 0: Checking Environment"
echo "========================================================"
# 检查预测结果目录是否存在
if [ ! -d "$PRED_DIR" ]; then
    echo "Error: Prediction directory not found at $PRED_DIR"
    exit 1
fi

# 检查目录下是否有 parquet 文件
if [ -z "$(ls -A $PRED_DIR/*.parquet 2>/dev/null)" ]; then
    echo "Error: No .parquet files found in $PRED_DIR. Please run generate.sh first."
    exit 1
fi

echo "========================================================"
echo "Eval Step 1: Locating Prediction Results"
echo "========================================================"
# 自动寻找该目录下最新生成的预测文件
LATEST_PRED=$(ls -t $PRED_DIR/*.parquet | head -n 1)
echo "Targeting latest prediction: $LATEST_PRED"

echo "========================================================"
echo "Eval Step 2: Fairness Evaluation (Query & Metrics)"
echo "========================================================"

# 运行评估脚本
python run_eval.py \
    --dataset $DATASET \
    --model_name "$MODEL_NAME" \
    --pred_path "$LATEST_PRED" \
    --query_attrs $QUERY_ATTRS

if [ $? -eq 0 ]; then
    echo "========================================================"
    echo "Evaluation Finished Successfully!"
    echo "Check your JSON report in: data/results/reports/"
    echo "========================================================"
else
    echo "Evaluation failed!"
    exit 1
fi