#!/bin/bash

# --- 配置区域 ---

DATASET="bbq_recognition"
MODEL_NAME="claude-3-5-sonnet-20241022"

# ✅ 新增：评估维度（1d / 2d / all）
EVAL_DIM="1d"

PRED_DIR="data/results/predictions/recognition"

# --- 执行区域 ---

echo "========================================================"
echo "Step 1: Locating Prediction Results"
echo "========================================================"

# 检查目录
if [ ! -d "$PRED_DIR" ]; then
    echo "❌ Error: Directory $PRED_DIR not found."
    exit 1
fi

# ✅ 只匹配对应 eval_dim 的 parquet
PATTERN="$PRED_DIR/${MODEL_NAME}_${EVAL_DIM}_*.parquet"
TARGET_PRED_FILES=$(ls $PATTERN 2>/dev/null)

if [ -z "$TARGET_PRED_FILES" ]; then
    echo "❌ Error: No prediction files found for:"
    echo "    Model = $MODEL_NAME"
    echo "    Eval Dim = $EVAL_DIM"
    echo "    Pattern = $PATTERN"
    exit 1
fi

# 取最新一个（按时间戳排序，字符串即可）
LATEST_PRED=$(echo "$TARGET_PRED_FILES" | sort | tail -n 1)

echo "Targeting latest prediction file:"
echo "  $LATEST_PRED"

echo "========================================================"
echo "Step 2: Running Recognition Evaluation"
echo "========================================================"

python run_eval_recognition.py \
    --dataset "$DATASET" \
    --pred_path "$LATEST_PRED" \
    --model_name "$MODEL_NAME" \
    --eval_dim "$EVAL_DIM"

if [ $? -eq 0 ]; then
    echo "========================================================"
    echo "✅ Evaluation Finished Successfully!"
    echo "Eval Dim: $EVAL_DIM"
    echo "Report saved in: data/results/reports/recognition/"
    echo "========================================================"
else
    echo "❌ Evaluation Failed!"
    exit 1
fi
