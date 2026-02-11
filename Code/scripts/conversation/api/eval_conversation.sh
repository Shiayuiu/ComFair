#!/bin/bash
set -euo pipefail

# --- 配置区域 ---
# MODEL_NAME="claude-3-5-sonnet-20241022"
# MODEL_NAME="claude-3-7-sonnet-20250219"
# MODEL_NAME="Deepseek-V3.2-reasoner"
# MODEL_NAME="Deepseek-V3.2-chat"
# MODEL_NAME="Gemini-3-Flash"
# MODEL_NAME="GPT-5-mini"
# MODEL_NAME="llama-3.1-8b-instruct"
# MODEL_NAME="llama-3.3-70b-instruct"
# MODEL_NAME="Qwen3-235B-A22B"
# MODEL_NAME="Qwen3-8B"，MODEL_NAME="Llama-3.1-8B-Instruct_base"，MODEL_NAME="Llama-3.1-8B-Instruct_augment"
# --- 任务配置 ---
DATASET="holistic_conversation"
MODEL_NAME="Llama-3.1-8B-Instruct_augment"
PRED_DIR="data/results/predictions/conversation"

# --- Judge (评估者) 配置 ---
# --- Judge (评估者) 配置 ---


JUDGE_MODEL_PATH="gpt-5.2" 
JUDGE_MODEL_NAME="gpt-5.2"
JUDGE_API_KEY=
JUDGE_BASE_URL= 

# --- 评估参数 ---
MAX_WORKERS=20

# （可选）只评估 1d/2d/all，如果你 run_eval_conversation.py 已加这个参数就能用
EVAL_DIM="1d"

echo "========================================================"
echo "Step 1: Locating Latest Conversation Results"
echo "========================================================"
echo "PRED_DIR: $PRED_DIR"
echo "MODEL_NAME filter: *${MODEL_NAME}*.parquet"

# 1) 收集候选文件（用数组更稳，避免空格/换行坑）
shopt -s nullglob
CANDIDATES=("$PRED_DIR"/*"$MODEL_NAME"*.parquet)
shopt -u nullglob

if [ ${#CANDIDATES[@]} -eq 0 ]; then
  echo "❌ Error: No prediction files found in $PRED_DIR matching *$MODEL_NAME*.parquet"
  echo ">>> Debug: list dir:"
  ls -lh "$PRED_DIR" | tail -n 50 || true
  exit 1
fi

# 2) 取“最新”：优先按文件修改时间（ls -t），再兜底按文件名排序
CANDIDATES=("$PRED_DIR"/*"$MODEL_NAME"*__"${EVAL_DIM}"__*.parquet)
LATEST_PRED="$(ls -t "${CANDIDATES[@]}" | head -n 1)"
if [ -z "$LATEST_PRED" ]; then
  # fallback：按文件名排序（适合 model_YYYYMMDD_HHMMSS.parquet）
  LATEST_PRED="$(printf '%s\n' "${CANDIDATES[@]}" | sort | tail -n 1)"
fi

echo "Targeting latest prediction file: $LATEST_PRED"

if [ -z "$LATEST_PRED" ] || [ ! -f "$LATEST_PRED" ]; then
  echo "❌ Error: Failed to locate latest parquet."
  exit 1
fi

# 3) 从文件名解析 model_name（保持你师兄写法，但更稳一点）
BASE_MODEL_NAME="$(basename "$LATEST_PRED" | sed -E 's/_[0-9]{8}_[0-9]{6}\.parquet$//')"
# 如果你的文件名不是 *_YYYYMMDD_HHMMSS.parquet，就退回 cut 的方式
if [ -z "$BASE_MODEL_NAME" ] || [ "$BASE_MODEL_NAME" = "$(basename "$LATEST_PRED")" ]; then
  BASE_MODEL_NAME="$(basename "$LATEST_PRED" | cut -d'_' -f1)"
fi

echo "Derived model_name for report: $BASE_MODEL_NAME"
echo "Using Judge: $JUDGE_MODEL_NAME"
echo "========================================================"
echo "Step 2: Running Judge-based Evaluation"
echo "========================================================"

# 执行评估
CMD=(python run_eval_conversation.py
  --dataset "$DATASET"
  --pred_path "$LATEST_PRED"
  --model_name "$BASE_MODEL_NAME"
  --judge_model_path "$JUDGE_MODEL_PATH"
  --judge_model_name "$JUDGE_MODEL_NAME"
  --judge_api_key "$JUDGE_API_KEY"
  --judge_base_url "$JUDGE_BASE_URL"
  --max_workers "$MAX_WORKERS"
)

# 如果你启用了 eval_dim，就取消注释下面两行
# if [ -n "${EVAL_DIM:-}" ]; then
#   CMD+=(--eval_dim "$EVAL_DIM")
# fi

echo ">>> Running:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"

echo "========================================================"
echo "✅ Evaluation Complete!"
echo "Parquet (scored): data/results/reports/conversation/*.parquet"
echo "JSON Report:      data/results/reports/conversation/*_report.json"
echo "Markdown Report:  outputs/conversation/*.md"
echo "========================================================"
