#!/bin/bash
set -euo pipefail

# ========================================================
# Configuration
# ========================================================

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
# MODEL_NAME="Qwen3-8B"
#MODEL_NAME="Llama-3.1-8B-Instruct_base"

# --- 任务配置 ---
DATASET="holistic_continuation"
MODEL_NAME="Llama-3.1-8B-Instruct_base"
PRED_DIR="data/results/predictions/continuation"

# --- Judge (评估者) 配置 ---
JUDGE_MODEL_PATH="gpt-5.2" 
JUDGE_MODEL_NAME="GPT-5.2"
JUDGE_API_KEY=
JUDGE_BASE_URL=

# --- 评估参数（推荐：快+稳） ---
MAX_WORKERS="${MAX_WORKERS:-10}"          # 8~16 推荐
EVAL_DIM="${EVAL_DIM:-1d}"              # 1d / 2d / all




JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-10}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-64}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
FLUSH_EVERY="${FLUSH_EVERY:-10}"
VERBOSE="${VERBOSE:-1}"

echo "========================================================"
echo "Step 1: Locating Latest Continuation Results (by filename dim tag)"
echo "========================================================"
echo "PRED_DIR: $PRED_DIR"
echo "MODEL_NAME: $MODEL_NAME"
echo "EVAL_DIM: $EVAL_DIM"
echo "Workers: $MAX_WORKERS | JudgeBatch: $JUDGE_BATCH_SIZE | JudgeMaxNew: $JUDGE_MAX_NEW_TOKENS | Timeout: ${REQUEST_TIMEOUT}s"

# --- 检查目录 ---
if [ ! -d "$PRED_DIR" ]; then
  echo "❌ Error: PRED_DIR not found: $PRED_DIR"
  exit 1
fi

shopt -s nullglob

# --- 按文件名维度筛选 ---
# 新命名规范：
#   <MODEL>__1d__YYYYMMDD_HHMMSS.parquet
#   <MODEL>__2d__YYYYMMDD_HHMMSS.parquet
if [ "$EVAL_DIM" = "1d" ] || [ "$EVAL_DIM" = "2d" ]; then
  PATTERN="$PRED_DIR/${MODEL_NAME}__${EVAL_DIM}__*.parquet"
  CANDIDATES=($PATTERN)
else
  # all：找该模型的所有 parquet（包含 __1d__/__2d__）
  PATTERN="$PRED_DIR/${MODEL_NAME}__*.parquet"
  CANDIDATES=($PATTERN)
fi

shopt -u nullglob

if [ ${#CANDIDATES[@]} -eq 0 ]; then
  echo "❌ Error: No prediction files found."
  echo "Expected pattern: $PATTERN"
  echo "Dir listing (tail):"
  ls -lh "$PRED_DIR" | tail -n 50 || true
  exit 1
fi

# --- 取最新（按修改时间）---
LATEST_PRED="$(ls -t "${CANDIDATES[@]}" 2>/dev/null | head -n 1 || true)"

if [ -z "$LATEST_PRED" ] || [ ! -f "$LATEST_PRED" ]; then
  echo "❌ Error: Failed to locate latest parquet."
  exit 1
fi

echo "Targeting latest prediction file: $LATEST_PRED"
echo "Using Judge: $JUDGE_MODEL_NAME ($JUDGE_MODEL_PATH)"

# --- 从文件名反推 model_name（去掉 __1d__/__2d__ + 时间戳）---
# 例：claude-3-5-sonnet-20241022__1d__20260128_143603.parquet -> claude-3-5-sonnet-20241022
BASE_MODEL_NAME="$(basename "$LATEST_PRED" | sed -E 's/__([12]d)__[0-9]{8}_[0-9]{6}\.parquet$//')"
if [ -z "$BASE_MODEL_NAME" ] || [ "$BASE_MODEL_NAME" = "$(basename "$LATEST_PRED")" ]; then
  # 兜底：找不到就用 MODEL_NAME
  BASE_MODEL_NAME="$MODEL_NAME"
fi
echo "Derived model_name for report: $BASE_MODEL_NAME"

echo "========================================================"
echo "Step 2: Running Judge-based Evaluation"
echo "========================================================"

CMD=(python run_eval_continuation.py
  --dataset "$DATASET"
  --pred_path "$LATEST_PRED"
  --model_name "$BASE_MODEL_NAME"
  --judge_model_path "$JUDGE_MODEL_PATH"
  --judge_model_name "$JUDGE_MODEL_NAME"
  --judge_api_key "$JUDGE_API_KEY"
  --judge_base_url "$JUDGE_BASE_URL"
  --max_workers "$MAX_WORKERS"
  --eval_dim "$EVAL_DIM"
  --judge_batch_size "$JUDGE_BATCH_SIZE"
  --judge_max_new_tokens "$JUDGE_MAX_NEW_TOKENS"
  --request_timeout "$REQUEST_TIMEOUT"
  --flush_every "$FLUSH_EVERY"
)

if [ "$VERBOSE" = "1" ]; then
  CMD+=(--verbose)
fi

echo ">>> Running:"
printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"

echo "========================================================"
echo "✅ Evaluation Complete!"
echo "Parquet (scored): data/results/reports/continuation/*.parquet"
echo "JSON Report:      data/results/reports/continuation/*_report.json"
echo "Markdown Report:  outputs/continuation/*.md"
echo "========================================================"
