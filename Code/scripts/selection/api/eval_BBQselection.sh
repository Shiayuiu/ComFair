#!/bin/bash
set -e

# ========================================================
# Configuration# --- 配置区域 ---
# MODEL_NAME="claude-3-5-sonnet-20241022"
# MODEL_NAME="claude-3-7-sonnet-20250219"
# MODEL_NAME="Deepseek-V3.2-reasoner"
# MODEL_NAME="Deepseek-V3.2-chat"
# MODEL_NAME="Gemini-3-Flash"
# MODEL_NAME="GPT-5-mini"
# MODEL_NAME="llama-3.1-8b-instruct"
# MODEL_NAME="llama-3.3-70b-instruct"
# MODEL_NAME="Qwen3-235B-A22B"
# MODEL_NAME="Qwen3-8B"
# ========================================================
DATASET="bbq_selection"
MODEL_NAME="Qwen3-8B"
PRED_DIR="data/results/predictions/selection"

# Evaluation dimension: 1d / 2d / all
EVAL_DIM="1d"

# ========================================================
# Step 1: Locate Latest Prediction File
# ========================================================
echo "========================================================"
echo "Step 1: Locating Latest Prediction Results"
echo "========================================================"

if [ ! -d "$PRED_DIR" ]; then
    echo "❌ Error: Directory not found: $PRED_DIR"
    exit 1
fi

# Match prediction files for this model + eval_dim
PATTERN="$PRED_DIR/${MODEL_NAME}_${EVAL_DIM}_*.parquet"

# List candidates (sorted by modification time, newest first)
CANDIDATES=$(ls -t $PATTERN 2>/dev/null || true)

if [ -z "$CANDIDATES" ]; then
    echo "❌ Error: No prediction files found."
    echo "    Model   = $MODEL_NAME"
    echo "    EvalDim = $EVAL_DIM"
    echo "    Pattern = $PATTERN"
    exit 1
fi

echo "Candidate prediction files (newest first):"
echo "$CANDIDATES" | sed 's/^/  - /'

# Pick the newest one
LATEST_PRED=$(echo "$CANDIDATES" | head -n 1)

echo ""
echo "Targeting latest prediction file:"
echo "  $LATEST_PRED"

# ========================================================
# Step 2: Run Selection Evaluation
# ========================================================
echo "========================================================"
echo "Step 2: Running Selection Evaluation"
echo "========================================================"

python run_eval_selection.py \
    --dataset "$DATASET" \
    --pred_path "$LATEST_PRED" \
    --model_name "$MODEL_NAME" \
    --eval_dim "$EVAL_DIM"

STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "========================================================"
    echo "✅ Evaluation Finished Successfully!"
    echo "Model    : $MODEL_NAME"
    echo "Eval Dim : $EVAL_DIM"
    echo "Report   : data/results/reports/selection/"
    echo "========================================================"
else
    echo "❌ Evaluation Failed!"
    exit $STATUS
fi
