import json
import os
import uuid
import glob
import random
from typing import List, Dict

class SelectionConcatenator:
    def __init__(self, raw_dir: str, output_dir: str, sample_size_per_file: int = 250, seed: int = 42):
        """
        Args:
            raw_dir: 原始 .jsonl 文件存放目录
            output_dir: 清洗合并后的输出目录
            sample_size_per_file: 每个原始文件采样的对数 (每对含 2 行)
            seed: 随机种子，确保实验可复现
        """
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.sample_size = sample_size_per_file
        self.seed = seed
        random.seed(self.seed)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _parse_filename(self, filename: str) -> Dict[str, str]:
        """解析文件名逻辑"""
        base_name = os.path.basename(filename).replace(".jsonl", "")
        parts = base_name.split("_")
        if len(parts) < 3:
            return None
        return {
            "base_attr": parts[0].lower(),
            "add_attr": parts[1].lower(),
            "add_attr_value": "_".join(parts[2:])
        }

    def process_and_concat(self):
        file_paths = glob.glob(os.path.join(self.raw_dir, "*.jsonl"))
        grouped_data = {}

        print(f">>> Found {len(file_paths)} raw files. Target sampling: {self.sample_size} pairs per file.")

        for path in file_paths:
            meta = self._parse_filename(path)
            if not meta:
                continue

            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            # 1. 将行组织成“对”
            pairs = []
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    pairs.append((lines[i], lines[i+1]))

            # 2. 执行采样逻辑
            total_pairs = len(pairs)
            if total_pairs < self.sample_size:
                print(f"  [Warning] {os.path.basename(path)} only has {total_pairs} pairs. Taking all.")
                sampled_pairs = pairs
            else:
                sampled_pairs = random.sample(pairs, self.sample_size)
                print(f"  [Sample] {os.path.basename(path)}: Sampled {self.sample_size} from {total_pairs} pairs.")

            # 3. 处理并存入内存分组
            group_key = (meta['base_attr'], meta['add_attr'])
            if group_key not in grouped_data:
                grouped_data[group_key] = []

            for base_str, aug_str in sampled_pairs:
                try:
                    base_row = json.loads(base_str)
                    aug_row = json.loads(aug_str)

                    

                    # 统一结构化
                    common_meta = {
                        "pair_id": base_row.get("source_id"),
                        "base_attr": meta['base_attr'].lower(),
                        "add_attr": meta['add_attr'].lower(),
                    }

                    # Selection任务特有字段
                    processed_base = {
                        **common_meta,
                        "sample_id": str(uuid.uuid4()),
                        "pair_type": "base",
                        "add_attr_value": "original",
                        "question": base_row["question"],  # 保留question字段
                        "ans0": base_row["ans0"],  # 保留ans0字段
                        "ans1": base_row["ans1"],  # 保留ans1字段
                        "ans2": base_row["ans2"],  # 保留ans2字段
                        "context": base_row["context"],
                        "label": base_row["label"]
                    }

                    processed_aug = {
                        **common_meta,
                        "sample_id": str(uuid.uuid4()),
                        "pair_type": "augmented",
                        "add_attr_value": meta['add_attr_value'],
                        "question": aug_row["question"],
                        "ans0": aug_row["ans0"],
                        "ans1": aug_row["ans1"],
                        "ans2": aug_row["ans2"],
                        "context": aug_row["context"],
                        "label": aug_row["label"]
                    }

                    grouped_data[group_key].extend([processed_base, processed_aug])
                except Exception as e:
                    print(f"  [Error] Failed to parse JSON in {path}: {e}")

        # 4. 写入合并后的文件
        self._save_grouped_files(grouped_data)

    def _save_grouped_files(self, grouped_data: Dict):
        for (base, add), records in grouped_data.items():
            output_filename = f"{base}_{add}.jsonl"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 每个合并文件的 sample_size 是该文件内所有记录的总和
            file_total_count = len(records)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for rec in records:
                    rec["sample_size"] = file_total_count
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            print(f">>> Exported: {output_path} (Total Records: {file_total_count})")

if __name__ == "__main__":
    RAW_INPUT = "./"  # 输入文件目录
    PROCESSED_OUTPUT = "./processed/"  # 输出目录
    
    # 设置每个文件采样 250 对
    concatenator = SelectionConcatenator(
        raw_dir=RAW_INPUT, 
        output_dir=PROCESSED_OUTPUT, 
        sample_size_per_file=250,
        seed=42
    )
    concatenator.process_and_concat()
