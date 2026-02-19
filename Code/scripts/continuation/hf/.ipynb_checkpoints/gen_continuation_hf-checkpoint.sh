#!/bin/bash

# --- 硬件配置 ---
export CUDA_VISIBLE_DEVICES=0

# --- 任务配置 ---
DATASET="holistic_continuation"
# 属性列表，脚本会扫描 data/processed/continuation/ 下匹配的文件
# 示例：会寻找 age_*.jsonl, race_*.jsonl 等
ATTRIBUTES="race gender"

# --- 模型配置 ---
MODEL_TYPE="hf"
MODEL_PATH="/path/to/your/model/"
MODEL_NAME="your-model-name"

# --- 推理参数 ---
BATCH_SIZE=4        # 续写任务较长，建议 Batch Size 设小一点以防 OOM
MAX_NEW_TOKENS=150  # 续写的长度
DEBUG_N=0           # 设置为 10 用于快速调试

echo "========================================================"
echo "Task: Continuation Generation - Local HF"
echo "Model: $MODEL_NAME"
echo "Attributes: $ATTRIBUTES"
echo "========================================================"

python run_gen_continuation.py \
    --dataset "$DATASET" \
    --attributes $ATTRIBUTES \
    --model_type "$MODEL_TYPE" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --debug_n "$DEBUG_N"

if [ $? -eq 0 ]; then
    echo "✅ Generation Finished! Results in data/results/predictions/continuation/"
else
    echo "❌ Generation Failed!"
    exit 1
fi