#!/bin/bash

# --- Configuration ---
# Evaluation task info
DATASET="adult"
MODEL_NAME="claude-3-7-sonnet-20250219"

# Query parameters (Query Stage)
# If empty (""), the script will use sensitive attributes defined in configs/[dataset].py
# To manually specify, e.g.: QUERY_ATTRS="race gender age"
QUERY_ATTRS="race relationship education"

# Paths
PRED_DIR="data/results/predictions/classification"

# --- Execution ---

echo "========================================================"
echo "Step 0: Checking Environment"
echo "========================================================"
# Verify prediction directory exists
if [ ! -d "$PRED_DIR" ]; then
    echo "Error: Prediction directory not found at $PRED_DIR"
    exit 1
fi

# Verify there are parquet files in the directory
if [ -z "$(ls -A $PRED_DIR/*.parquet 2>/dev/null)" ]; then
    echo "Error: No .parquet files found in $PRED_DIR. Please run the generation script first."
    exit 1
fi

echo "========================================================"
echo "Step 1: Locating Prediction Results"
echo "========================================================"

# Auto-find the latest prediction file for the model
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

# Append query attributes if specified
if [ ! -z "$QUERY_ATTRS" ]; then
    EVAL_CMD="$EVAL_CMD --query_attrs $QUERY_ATTRS"
fi

# Run evaluation script
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