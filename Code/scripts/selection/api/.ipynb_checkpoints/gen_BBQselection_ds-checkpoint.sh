#!/bin/bash
set -e

# ========================================================
# Configuration
# ========================================================
DATASET="bbq_selection"

# Evaluation dimension: 1d / 2d / all
EVAL_DIM="2d"

# ✅ 主轴（base attributes）
BASE_ATTRS="age race gender"


#  注入维度精确（不写为全选）
#  inject_attrs=""


# 模型配置
# 'api' (GPT-4, Llama-API) 或 'reasoning_api' (o1, R1)
MODEL_TYPE="api"
MODEL_PATH="deepseek-chat" 
MODEL_NAME="Deepseek-V3.2-chat"

# API 密钥与地址 
API_KEY="sk-dc8715c8551040a3848863016300e39a"
BASE_URL="https://api.deepseek.com/v1" 

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
