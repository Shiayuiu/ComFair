#!/bin/bash
set -euo pipefail

# ==============================
# Task config
# ==============================
DATASET="holistic_continuation"
ATTRIBUTES=(age gender race)

EVAL_DIM="1d"
BASE_ATTRS=(race)
# INJECT_ATTRS=(gender race)   # 可选：只跑特定注入组合（顺序无关，脚本会原样传参）
# --- 模型配置 ---
MODEL_TYPE="reasoning_api"
MODEL_PATH="deepseek-reasoner" 
MODEL_NAME="Deepseek-V3.2-reasoner"

BASE_URL= # 替换为实际的API URL
API_KEY="your_api_key_here" # 替换为你的API Key



# ==============================
# Inference parameters (anti-stuck)
# ==============================
MAX_WORKERS="${MAX_WORKERS:-20}"        #
BATCH_SIZE="${BATCH_SIZE:-1}"         
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-800}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"  
FLUSH_EVERY="${FLUSH_EVERY:-1}"          
DEBUG_N="${DEBUG_N:-0}"

# verbose 日志：每批都会打印正在请求第几批
VERBOSE_FLAG="--verbose"

echo "========================================================"
echo "Task: Continuation Generation - API (2D)"
echo "Model: ${MODEL_NAME} (${MODEL_TYPE})"
echo "Eval Dim: ${EVAL_DIM}"
echo "Attributes: ${ATTRIBUTES[*]}"
echo "Base Attrs: ${BASE_ATTRS[*]}"
echo "Inject Attrs: ${INJECT_ATTRS[*]:-<none>}"
echo "Base URL: ${BASE_URL}"
echo "Workers: ${MAX_WORKERS} | Batch: ${BATCH_SIZE} | MaxNewTokens: ${MAX_NEW_TOKENS}"
echo "Timeout: ${REQUEST_TIMEOUT}s | FlushEvery: ${FLUSH_EVERY} | DebugN: ${DEBUG_N}"
echo "========================================================"

CMD=(python run_gen_continuation.py
  --dataset "${DATASET}"
  --attributes "${ATTRIBUTES[@]}"
  --eval_dim "${EVAL_DIM}"
  --base_attrs "${BASE_ATTRS[@]}"
  --model_type "${MODEL_TYPE}"
  --model_name "${MODEL_NAME}"
  --model_path "${MODEL_PATH}"
  --api_key "${API_KEY}"
  --base_url "${BASE_URL}"
  --max_workers "${MAX_WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --request_timeout "${REQUEST_TIMEOUT}"
  --flush_every "${FLUSH_EVERY}"
  --debug_n "${DEBUG_N}"
  ${VERBOSE_FLAG}
)

if [ -n "${INJECT_ATTRS[*]:-}" ]; then
  CMD+=(--inject_attrs "${INJECT_ATTRS[@]}")
fi

echo ">>> Running:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"

echo "✅ API Generation Finished!"
echo "Results saved in: data/results/predictions/continuation/"
