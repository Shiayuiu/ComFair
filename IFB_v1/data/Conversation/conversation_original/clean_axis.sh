#!/bin/bash
# CEB数据axis字段清洗脚本
# 将axis字段从旧格式转换为新格式

set -e  # 出错时退出

# ============================================
# 1. 参数配置
# ============================================
INPUT_FILE="gender_race_black.jsonl"              # 输入文件路径
OUTPUT_FILE="gender_rece_black_clean.jsonl"            # 输出文件路径
BACKUP_DIR="backup"                   # 备份目录

# ============================================
# 2. 创建备份
# ============================================
echo "创建输入文件备份..."
if [ ! -d "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
fi

BACKUP_FILE="${BACKUP_DIR}/$(basename "$INPUT_FILE")_backup_$(date +%Y%m%d_%H%M%S).jsonl"
cp "$INPUT_FILE" "$BACKUP_FILE"
echo "备份已创建: $BACKUP_FILE"

# ============================================
# 3. 清洗函数
# ============================================
clean_axis_field() {
    local input_file="$1"
    local output_file="$2"
    
    echo "开始清洗axis字段..."
    
    # 使用Python进行精确的JSON处理
    python3 -c "
import json
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

transformations = 0
errors = 0
total_lines = 0

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line_num, line in enumerate(infile, 1):
        line = line.strip()
        if not line:
            continue
            
        total_lines += 1
        
        try:
            data = json.loads(line)
            
            # 检查并转换axis字段
            if 'axis' in data:
                old_value = data['axis']
                new_value = None
                
                # 精确匹配转换
                if old_value == 'race_ethnicity':
                    new_value = 'race'
                elif old_value == 'gender_and_sex':
                    new_value = 'gender'
                elif old_value == 'age':
                    new_value = 'age'  # 保持不变
                else:
                    # 其他值保持不变
                    new_value = old_value
                
                # 如果值发生变化，记录并更新
                if new_value != old_value:
                    data['axis'] = new_value
                    transformations += 1
                    print(f'行 {line_num}: {old_value} -> {new_value}')
            
            # 写入清洗后的数据
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            
        except json.JSONDecodeError as e:
            errors += 1
            print(f'错误: 行 {line_num} JSON解析失败: {e}')
            # 保持原行不变（写入错误行）
            outfile.write(line + '\n')
        except Exception as e:
            errors += 1
            print(f'错误: 行 {line_num} 处理失败: {e}')
            # 保持原行不变
            outfile.write(line + '\n')

print(f'处理完成!')
print(f'总行数: {total_lines}')
print(f'转换次数: {transformations}')
print(f'错误行数: {errors}')
" "$input_file" "$output_file"
}

# ============================================
# 4. 执行清洗
# ============================================
echo "=========================================="
echo "开始清洗数据..."
echo "- 输入文件: $INPUT_FILE"
echo "- 输出文件: $OUTPUT_FILE"
echo "=========================================="

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    exit 1
fi

# 执行清洗
clean_axis_field "$INPUT_FILE" "$OUTPUT_FILE"

# ============================================
# 5. 验证结果
# ============================================
echo "=========================================="
echo "验证清洗结果..."
echo "=========================================="

# 统计原始和清洗后的axis值分布
echo "原始文件axis值分布:"
python3 -c "
import json
from collections import Counter
import sys

filename = sys.argv[1]
counter = Counter()

with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            if 'axis' in data:
                counter[data['axis']] += 1
        except:
            pass

for value, count in counter.most_common():
    print(f'  {value}: {count}')
" "$BACKUP_FILE"

echo "清洗后文件axis值分布:"
python3 -c "
import json
from collections import Counter
import sys

filename = sys.argv[1]
counter = Counter()

with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            if 'axis' in data:
                counter[data['axis']] += 1
        except:
            pass

for value, count in counter.most_common():
    print(f'  {value}: {count}')
" "$OUTPUT_FILE"

# ============================================
# 6. 完整性检查
# ============================================
echo "=========================================="
echo "完整性检查..."
echo "=========================================="

# 检查行数是否一致
original_lines=$(wc -l < "$BACKUP_FILE")
cleaned_lines=$(wc -l < "$OUTPUT_FILE")

echo "原始文件行数: $original_lines"
echo "清洗后文件行数: $cleaned_lines"

if [ "$original_lines" -eq "$cleaned_lines" ]; then
    echo "✓ 行数一致"
else
    echo "警告: 行数不一致!"
    echo "差异: $((cleaned_lines - original_lines)) 行"
fi

# 检查文件大小
original_size=$(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE")
cleaned_size=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE")

echo "原始文件大小: $original_size 字节"
echo "清洗后文件大小: $cleaned_size 字节"

# ============================================
# 7. 可选：替换原始文件
# ============================================
read -p "是否用清洗后的文件替换原始文件? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cp "$OUTPUT_FILE" "$INPUT_FILE"
    echo "已替换原始文件: $INPUT_FILE"
fi

echo "=========================================="
echo "清洗完成!"
echo "备份文件: $BACKUP_FILE"
echo "清洗结果: $OUTPUT_FILE"
echo "=========================================="