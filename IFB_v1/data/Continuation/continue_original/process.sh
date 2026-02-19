#!/bin/bash
# CEB数据清洗和属性注入脚本
# 使用方法: ./run_ceb_clean.sh

set -e  # 出错时退出

# ============================================
# 1. 必需参数配置
# ============================================
BASE_ATTR="gender"                       # 基础属性维度 (从axis字段获取，用于输入文件名)
ADD_ATTR="race"                     # 要添加的属性维度 (age/gender/race/religion)
ADD_ATTR_VALUE="white"               # 要添加的属性值 (如 female, young, asian 等)

# 基于参数生成文件名
INPUT_FILE="${BASE_ATTR}.jsonl"       # 输入文件: [BASE_ATTR].jsonl
OUTPUT_FILE="${BASE_ATTR,,}_${ADD_ATTR,,}_${ADD_ATTR_VALUE,,}.jsonl"  # 输出文件: [base_attr]_[add_attr]_[add_attr_value].jsonl (全小写)

# ============================================
# 2. API配置 (必需)
# ============================================
API_KEY="sk-dc8715c8551040a3848863016300e39a"   # 替换为你的API密钥
BASE_URL="https://api.deepseek.com/v1"          # 留空则使用默认OpenAI API
# 或使用环境变量:
# export OPENAI_API_KEY="sk-..."
# export YUNAI_API_KEY="sk-..."

# ============================================
# 3. 模型配置 (推荐值)
# ============================================
INJECT_MODEL="deepseek-chat"           # 属性注入模型
QC_MODEL="deepseek-chat"              # 质量检查模型

# ============================================
# 4. 性能参数
# ============================================
WORKERS=20                           # 并发工作线程数 (推荐: 4-8，根据API限制调整)
RETRIES=5                            # API失败重试次数
MAX_TOKENS_INJECT=500                # 注入API最大token数
MAX_TOKENS_QC=100                    # QC API最大token数

# ============================================
# 5. 温度参数
# ============================================
TEMP_INJECT=0.0                      # 注入温度 (0.0确保确定性输出)
TEMP_QC=0.0                          # QC温度 (0.0确保确定性输出)

# ============================================
# 6. 间隔参数 (避免API限流)
# ============================================
SLEEP_INJECT=0.1                     # 注入API调用间隔(秒)
SLEEP_QC=0.1                         # QC API调用间隔(秒)

# ============================================
# 7. 处理控制参数
# ============================================
LIMIT=0                              # 限制处理行数 (0表示处理所有)
LOG_EVERY=20                         # 日志打印频率 (每处理N行打印一次)
DEBUG_API=0                          # API调试模式 (1开启，0关闭)

# ============================================
# 8. 文件检查
# ============================================
echo "检查输入文件..."
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    exit 1
fi

# ============================================
# 10. 执行Python脚本
# ============================================
echo "开始处理数据..."
echo "=========================================="
echo "配置参数:"
echo "- 输入文件:    $INPUT_FILE"
echo "- 输出文件:    $OUTPUT_FILE"
echo "- 添加属性:    $ADD_ATTR"
echo "- 属性值:      $ADD_ATTR_VALUE"
echo "- 注入模型:    $INJECT_MODEL"
echo "- QC模型:      $QC_MODEL"
echo "- 工作线程:    $WORKERS"
echo "- 处理行数:    ${LIMIT:-全部}"
echo "=========================================="

python3 process.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --add_attr "$ADD_ATTR" \
    --add_attr_value "$ADD_ATTR_VALUE" \
    --inject_model "$INJECT_MODEL" \
    --qc_model "$QC_MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --workers "$WORKERS" \
    --retries "$RETRIES" \
    --max_tokens_inject "$MAX_TOKENS_INJECT" \
    --max_tokens_qc "$MAX_TOKENS_QC" \
    --temp_inject "$TEMP_INJECT" \
    --temp_qc "$TEMP_QC" \
    --sleep_inject "$SLEEP_INJECT" \
    --sleep_qc "$SLEEP_QC" \
    --limit "$LIMIT" \
    --log_every "$LOG_EVERY" \
    --debug_api "$DEBUG_API"

# ============================================
# 11. 结果统计
# ============================================
echo "=========================================="
echo "处理完成!"
echo "输出文件: $OUTPUT_FILE"

if [ -f "$OUTPUT_FILE" ]; then
    LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
    echo "输出文件行数: $LINE_COUNT"
    
    if [ $LINE_COUNT -gt 0 ]; then
        # 显示前两行作为示例
        echo "输出示例 (前2行):"
        echo "------------------------------------------"
        head -n 2 "$OUTPUT_FILE" | while read line; do
            echo "$line"
        done
        echo "------------------------------------------"
        
        # 检查是否是偶数行
        if [ $((LINE_COUNT % 2)) -eq 0 ]; then
            PAIR_COUNT=$((LINE_COUNT / 2))
            echo "生成数据对数量: $PAIR_COUNT"
        else
            echo "警告: 输出文件行数不是偶数，可能有问题"
        fi
    fi
fi

echo "=========================================="
