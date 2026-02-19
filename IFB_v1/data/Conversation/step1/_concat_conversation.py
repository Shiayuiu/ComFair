import json
import os
import uuid
import glob
import re
import argparse
from typing import Dict

class ConversationConcatenator:
    def __init__(self, raw_dir: str, output_dir: str):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _parse_filename(self, filename: str) -> Dict[str, str]:
        """
        解析文件名: [base_attr]_[add_attr]_[add_attr_value].jsonl
        例如: age_gender_female.jsonl
        """
        base_name = os.path.basename(filename).replace(".jsonl", "")
        parts = base_name.split("_")
        if len(parts) < 3:
            return None
        return {
            "base_attr": parts[0],
            "add_attr": parts[1],
            "add_attr_value": "_".join(parts[2:])
        }

    def _extract_conversation_text(self, prompt: str) -> str:
        """
        从 Prompt 中提取双引号内的对话内容。
        Prompt 示例: ... beginning of a conversation: "Person A: ... \nPerson B: ..." Can you ...
        """
        # 使用 DOTALL 模式匹配换行符
        match = re.search(r'conversation: "(.*?)"', prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def process_and_concat(self):
        file_paths = glob.glob(os.path.join(self.raw_dir, "*.jsonl"))
        grouped_data = {}

        print(f">>> Found {len(file_paths)} raw files in {self.raw_dir}")

        for path in file_paths:
            meta = self._parse_filename(path)
            if not meta:
                print(f"  [Skip] Invalid filename: {path}")
                continue

            # 分组 Key: (age, gender)
            group_key = (meta['base_attr'], meta['add_attr'])
            if group_key not in grouped_data:
                grouped_data[group_key] = []

            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        raw_row = json.loads(line.strip())
                        
                        # 1. 解析原始 ID (提取数字部分)
                        # "8" -> "8", "8_inject_female" -> "8"
                        raw_id_str = str(raw_row.get("id", ""))
                        id_num = raw_id_str.split("_")[0] 
                        
                        # 2. 构造 Pair ID (数字ID + 属性值)
                        # 确保来自不同文件的同源 ID 能配对，但不同属性的 ID 互不干扰
                        # e.g. "8_female"
                        pair_id = f"{id_num}_{meta['add_attr_value']}"

                        # 3. 提取对话文本
                        text_content = self._extract_conversation_text(raw_row.get("prompt", ""))
                        if not text_content:
                            text_content = raw_row.get("prompt", "") # Fallback

                        # 4. 判断 Pair Type 和 Bucket
                        bucket_data = raw_row.get("bucket")
                        if isinstance(bucket_data, dict) and "inject" in bucket_data:
                            pair_type = "augmented"
                            bucket_label = bucket_data.get("original", "unknown")
                        else:
                            pair_type = "base"
                            bucket_label = bucket_data if isinstance(bucket_data, str) else "unknown"

                        # 5. 构造标准 Record
                        clean_record = {
                            "sample_id": str(uuid.uuid4()),
                            "pair_id": pair_id,
                            "pair_type": pair_type,
                            "base_attr": meta['base_attr'],
                            "add_attr": meta['add_attr'],
                            "add_attr_value": meta['add_attr_value'], # e.g. "female" or "original"
                            "bucket_label": bucket_label,
                            "text": text_content
                        }
                        
                        grouped_data[group_key].append(clean_record)

                    except json.JSONDecodeError:
                        print(f"  [Error] Bad JSON in {path}")

        # 写入合并文件
        self._save_grouped_files(grouped_data)

    def _save_grouped_files(self, grouped_data: Dict):
        for (base, add), records in grouped_data.items():
            output_filename = f"{base}_{add}.jsonl"
            output_path = os.path.join(self.output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            print(f">>> Exported: {output_path} (Records: {len(records)})")

if __name__ == "__main__":
    # 请确保创建了 data/raw/conversation 目录并放入了原始数据
    RAW_INPUT = "./IFB_v1/data/Conversation/step1/"
    PROCESSED_OUTPUT = "./Code/data/processed/conversation/"
    
    concatenator = ConversationConcatenator(RAW_INPUT, PROCESSED_OUTPUT)
    concatenator.process_and_concat()