#!/bin/bash

# 遍历当前目录下的所有文件和目录
for file in *; do
    # 跳过目录和脚本本身
    if [[ -d "$file" ]] || [[ "$file" == "rename.sh" ]]; then
        continue
    fi
    
    # 获取文件名和扩展名
    filename="$file"
    extension=""
    
    # 检查是否有扩展名
    if [[ "$filename" =~ \.jsonl$ ]]; then
        extension=".jsonl"
        filename="${filename%.jsonl}"
    fi
    
    # 按"."分割文件名
    IFS='.' read -ra parts <<< "$filename"
    
    # 检查是否有足够的部分
    if [[ ${#parts[@]} -ge 3 ]]; then
        # 获取第一部分（如 age 或 Gender_identity）
        first_part="${parts[0]}"
        
        # 获取第三部分并去掉开头的"plus_"
        second_part="${parts[1]}"
        if [[ "$second_part" == plus_* ]]; then
            second_part="${second_part#plus_}"
        fi
        
        # 提取第一部分的第一个单词（直到下划线或结尾）
        # 例如: "Gender_identity" -> "Gender", "age" -> "age"
        if [[ "$first_part" =~ _ ]]; then
            first_part="${first_part%%_*}"
        fi
        
        # 构建新文件名
        new_filename="${first_part}_${second_part}${extension}"
        
        # 重命名文件（先显示，确认无误后取消注释下面的mv命令）
        echo "重命名: '$file' -> '$new_filename'"
        mv "$file" "$new_filename"
    else
        echo "跳过 '$file' - 格式不符合要求"
    fi
done

echo "完成！要实际执行重命名，请取消脚本中mv命令的注释。"