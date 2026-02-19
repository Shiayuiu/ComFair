#!/bin/bash

# --- 配置区域 ---
# 显卡设置
export CUDA_VISIBLE_DEVICES=2

# 任务参数
DATASET="bbq_recognition"
# 指定要作为 base_attr 生成的属性列表 (对应文件名中的前缀)
# 例如: "age gender status race" 会寻找 age_*.jsonl, gender_*.jsonl 等
ATTRIBUTES="gender race"

# 模型配置
MODEL_TYPE="hf"
# 修改为你的本地模型路径
MODEL_PATH="/zhdd/home/yfwu/model/Qwen2.5-1.5B-Instruct/"
MODEL_NAME="Qwen3-8B"

# 推理参数
BATCH_SIZE=10
DEBUG_N=10  # 设为 >0 (如 10) 可进行快速测试

# --- 执行区域 ---

echo "========================================================"
echo "Task: Recognition (BBQ-Augmented) - Local Model"
echo "Model: $MODEL_NAME"
echo "Attributes: $ATTRIBUTES"
echo "========================================================"

# 运行生成脚本
python run_gen_recognition.py \
    --dataset "$DATASET" \
    --attributes $ATTRIBUTES \
    --model_type "$MODEL_TYPE" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --debug_n "$DEBUG_N"

if [ $? -eq 0 ]; then
    echo "Generation Finished!"
    echo "Results saved in: data/results/predictions/recognition/"
else
    echo "Generation Failed!"
    exit 1
fi