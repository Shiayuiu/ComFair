#!/bin/bash
set -e

# ========================================================
# Configuration
# ========================================================
DATASET="bbq_recognition"

# Evaluation dimension: 1d / 2d / all
EVAL_DIM="1d"

# ✅ 主轴（base attributes）——识别文件按这个来匹配
BASE_ATTRS="race"

# 模型配置
# 选项: 'api' (普通LLM) 或 'reasoning_api' (o1, R1 等推理模型)
MODEL_TYPE="api"
# API 中的模型 ID
MODEL_PATH="deepseek-chat"
MODEL_NAME="Deepseek-V3.2-chat"

# API 密钥与地址（✅ bash 里用普通双引号）
BASE_URL= # 替换为实际的API URL
API_KEY="your_api_key_here" # 替换为你的API Key

BATCH_SIZE=10
MAX_WORKERS=100
DEBUG_N=0

# ========================================================
# Info
# ========================================================
echo "========================================================"
echo "Task      : Recognition (BBQ)"
echo "Model     : ${MODEL_NAME} (${MODEL_TYPE})"
echo "Eval Dim  : ${EVAL_DIM}"
echo "Base Attrs: ${BASE_ATTRS}"
echo "Inject    : unrestricted (file-driven)"
echo "========================================================"

# ========================================================
# Run Generation
# ========================================================
python run_gen_recognition.py \
  --dataset "${DATASET}" \
  --eval_dim "${EVAL_DIM}" \
  --base_attrs ${BASE_ATTRS} \
  --model_type "${MODEL_TYPE}" \
  --model_name "${MODEL_NAME}" \
  --model_path "${MODEL_PATH}" \
  --api_key "${API_KEY}" \
  --base_url "${BASE_URL}" \
  --batch_size "${BATCH_SIZE}" \
  --max_workers "${MAX_WORKERS}" \
  --debug_n "${DEBUG_N}"

STATUS=$?

if [ ${STATUS} -eq 0 ]; then
  echo "========================================================"
  echo "✅ Generation Finished Successfully!"
  echo "Results saved in: data/results/predictions/recognition/"
  echo "========================================================"
else
  echo "❌ Generation Failed!"
  exit ${STATUS}
fi
