#!/bin/bash

# --- 任务配置 ---
DATASET="holistic_conversation"
ATTRIBUTES="age gender race"     # ✅建议写全，避免 base_attrs 过滤漏掉

# --- 2D 配置 ---
EVAL_DIM="1d"
BASE_ATTRS="race"     # ✅你要评哪些主轴就写哪些
# INJECT_ATTRS="gender race"     # （可选）只跑某一种 2D 注入组合，取消注释启用

# --- 模型配置 ---
MODEL_TYPE="api"
MODEL_PATH="gpt-5-mini"
MODEL_NAME="GPT-5-mini"

# --- API 凭证 ---
BASE_URL= # 替换为实际的API URL
API_KEY="your_api_key_here" # 替换为你的API Key

# --- 推理参数 ---
MAX_WORKERS=20
MAX_NEW_TOKENS=1000
DEBUG_N=0

echo "========================================================"
echo "Task: Conversation Generation - API (2D)"
echo "Model: $MODEL_NAME ($MODEL_TYPE)"
echo "Eval Dim: $EVAL_DIM"
echo "Attributes: $ATTRIBUTES"
echo "Base Attrs: $BASE_ATTRS"
echo "Inject Attrs: ${INJECT_ATTRS:-<none>}"
echo "========================================================"

CMD=(python run_gen_conversation.py
    --dataset "$DATASET"
    --attributes $ATTRIBUTES
    --eval_dim "$EVAL_DIM"
    --base_attrs $BASE_ATTRS
    --model_type "$MODEL_TYPE"
    --model_name "$MODEL_NAME"
    --model_path "$MODEL_PATH"
    --api_key "$API_KEY"
    --base_url "$BASE_URL"
    --max_workers "$MAX_WORKERS"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --debug_n "$DEBUG_N"
)

# （可选）注入属性精确匹配过滤：集合匹配 set(injects)==set(inject_attrs)
if [ -n "$INJECT_ATTRS" ]; then
  CMD+=(--inject_attrs $INJECT_ATTRS)
fi

echo ">>> Running:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"
RET=$?

if [ $RET -eq 0 ]; then
    echo "✅ API Generation Finished!"
    echo "Results saved in: data/results/predictions/conversation/"
else
    echo "❌ API Generation Failed! (exit=$RET)"
    exit $RET
fi
