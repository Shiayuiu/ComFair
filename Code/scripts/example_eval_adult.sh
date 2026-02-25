#!/bin/bash

# -------------------------------
# Configuration
# -------------------------------

# Basic evaluation settings
DATASET="adult"
MODEL_NAME="Qwen2.5-1.5B"

# Query stage attributes.
# If empty (""), the script will use default sensitive attributes
# defined in configs/[dataset].py.
# Example manual setting: QUERY_ATTRS="race gender age"
QUERY_ATTRS="race gender age"

# Prediction results directory
PRED_DIR="./data/results/predictions/classification"

# -------------------------------
# Execution
# -------------------------------

echo "========================================================"
echo "Step 0: Checking Environment"
echo "========================================================"

# Check if prediction directory exists
if [ ! -d "$PRED_DIR" ]; then
    echo "Error: Prediction directory not found at $PRED_DIR"
    exit 1
fi

# Check if any parquet files exist in the directory
if [ -z "$(ls -A $PRED_DIR/*.parquet 2>/dev/null)" ]; then
    echo "Error: No .parquet files found in $PRED_DIR. Please run the generation script first."
    exit 1
fi

echo "========================================================"
echo "Step 1: Locating Prediction Results"
echo "========================================================"

# Automatically select the most recently generated prediction file
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

# Execute evaluation script
python run_eval.py \
    --dataset $DATASET \
    --model_name "$MODEL_NAME" \
    --pred_path "$LATEST_PRED" \
    --query_attrs $QUERY_ATTRS

if [ $? -eq 0 ]; then
    echo "========================================================"
    echo "Evaluation Finished Successfully!"
    echo "Check your JSON report in: data/results/reports/classification/"
    echo "========================================================"
else
    echo "Evaluation failed!"
    exit 1
fi