#!/bin/bash

# -------------------------------
# Configuration
# -------------------------------

DATASET="bbq_recognition"
MODEL_NAME="YOUR_MODEL_NAME"

# Evaluation dimension: 1d / 2d / all
EVAL_DIM="1d"

PRED_DIR="data/results/predictions/recognition"

# -------------------------------
# Execution
# -------------------------------

echo "========================================================"
echo "Step 1: Locating Prediction Results"
echo "========================================================"

# Check if prediction directory exists
if [ ! -d "$PRED_DIR" ]; then
    echo "Error: Directory $PRED_DIR not found."
    exit 1
fi

# Match parquet files corresponding to the specified evaluation dimension
PATTERN="$PRED_DIR/${MODEL_NAME}_${EVAL_DIM}_*.parquet"
TARGET_PRED_FILES=$(ls $PATTERN 2>/dev/null)

if [ -z "$TARGET_PRED_FILES" ]; then
    echo "Error: No prediction files found for:"
    echo "  Model = $MODEL_NAME"
    echo "  Eval Dim = $EVAL_DIM"
    echo "  Pattern = $PATTERN"
    exit 1
fi

# Select the most recent file (sorted by timestamp in filename)
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
    echo "Evaluation Finished Successfully!"
    echo "Eval Dim: $EVAL_DIM"
    echo "Report saved in: data/results/reports/recognition/"
    echo "========================================================"
else
    echo "Evaluation Failed!"
    exit 1
fi