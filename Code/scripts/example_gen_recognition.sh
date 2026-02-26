#!/bin/bash
set -e

# ========================================================
# Configuration
# ========================================================

DATASET="bbq_recognition"

# Evaluation dimension: 1d / 2d / all
EVAL_DIM="1d"

# Base attributes used for matching recognition files
BASE_ATTRS="race"

# Model configuration
# Options: 'api' (standard LLM) or 'reasoning_api' (reasoning models such as o1 or R1)
MODEL_TYPE="api"

# Model identifier (API-side model ID)
MODEL_PATH="YOUR_MODEL_IDENTIFIER"
MODEL_NAME="YOUR_MODEL_NAME"

# API configuration
BASE_URL="YOUR_API_BASE_URL"
API_KEY="YOUR_API_KEY"

BATCH_SIZE=10
MAX_WORKERS=100

# Run N samples, set DEBUG_N=0 to run all samples
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
  echo "Generation Finished Successfully!"
  echo "Results saved in: data/results/predictions/recognition/"
  echo "========================================================"
else
  echo "Generation Failed!"
  exit ${STATUS}
fi