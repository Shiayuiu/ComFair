#!/bin/bash

# --- 配置区域 ---
# 评估任务基本信息
DATASET="adult"
MODEL_NAME="Qwen3-8B"

# 查询参数 (Query Stage)
# 如果留空 ""，则脚本会调用 configs/[dataset].py 中默认定义的敏感属性
# 如果需要手动指定，例如: QUERY_ATTRS="race gender age"
QUERY_ATTRS="race gender age"

# 结果路径配置
PRED_DIR="./data/results/predictions/"

# --- 执行区域 ---

echo "========================================================"
echo "Step 0: Checking Environment"
echo "========================================================"
# 检查预测结果目录是否存在
if [ ! -d "$PRED_DIR" ]; then
    echo "Error: Prediction directory not found at $PRED_DIR"
    exit 1
fi

# 检查目录下是否有 parquet 文件
if [ -z "$(ls -A $PRED_DIR/*.parquet 2>/dev/null)" ]; then
    echo "Error: No .parquet files found in $PRED_DIR. Please run generate script first."
    exit 1
fi

echo "========================================================"
echo "Step 1: Locating Prediction Results"
echo "========================================================"

# 自动寻找该目录下最新生成的预测文件
TARGET_PRED_FILES=$(ls $PRED_DIR/*$MODEL_NAME*.parquet 2>/dev/null)
LATEST_PRED=$(echo "$TARGET_PRED_FILES" | sort -t '-' -k2,2r | head -n 1)
echo "Targeting latest prediction file: $LATEST_PRED"

if [ -z "$TARGET_PRED_FILES" ]; then
    echo "Error: No prediction files found for model $MODEL_NAME."
    exit 1
fi
echo "========================================================"
echo "Step 2: Fairness Evaluation (Query & Metrics)"
echo "========================================================"

# 如果用户指定了查询属性，则追加参数
if [ ! -z "$QUERY_ATTRS" ]; then
    EVAL_CMD="$EVAL_CMD --query_attrs $QUERY_ATTRS"
fi

# 运行评估脚本
# eval $EVAL_CMD
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