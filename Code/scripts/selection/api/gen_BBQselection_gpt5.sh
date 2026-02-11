#!/bin/bash
set -e

# ========================================================
# Configuration
# ========================================================
DATASET="bbq_selection"

# Evaluation dimension: 1d / 2d / all
EVAL_DIM="1d"

# ✅ 主轴（base attributes）
BASE_ATTRS="race"


#  注入维度精确（不写为全选）
#  inject_attrs=""


# 模型配置

MODEL_TYPE="api"
MODEL_PATH="gpt-5-mini"
MODEL_NAME="GPT-5-mini"

# API 配置（建议用环境变量，这里仅示例）
BASE_URL= # 替换为实际的API URL
API_KEY="your_api_key_here" # 替换为你的API Key

BATCH_SIZE=10
MAX_WORKERS=100
DEBUG_N=0

# ========================================================
# Info
# ========================================================
echo "========================================================"
echo "Task: Selection (BBQ) - API Model"
echo "Model     : $MODEL_NAME ($MODEL_TYPE)"
echo "Eval Dim  : $EVAL_DIM"
echo "Base Attrs: $BASE_ATTRS"
echo "Inject    : unrestricted (file-driven)"
echo "========================================================"

# ========================================================
# Run Generation
# ========================================================
python run_gen_selection.py \
  --dataset "$DATASET" \
  --eval_dim "$EVAL_DIM" \
  --base_attrs $BASE_ATTRS \
  --model_type "$MODEL_TYPE" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --api_key "$API_KEY" \
  --base_url "$BASE_URL" \
  --batch_size "$BATCH_SIZE" \
  --max_workers "$MAX_WORKERS" \
  --debug_n "$DEBUG_N"

STATUS=$?

if [ $STATUS -eq 0 ]; then
  echo "========================================================"
  echo "✅ Generation Finished Successfully!"
  echo "Results saved in: data/results/predictions/selection/"
  echo "========================================================"
else
  echo "❌ Generation Failed!"
  exit $STATUS
fi
