import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

def fix_parquet_pair_ids(file_path):
    try:
        # 1. 读取结果文件
        df = pd.read_parquet(file_path)
        
        # 检查是否为空
        if len(df) == 0:
            print(f"  [Skip] Empty file: {os.path.basename(file_path)}")
            return

        # 2. 验证行数是否为偶数 (基本的完整性检查)
        if len(df) % 2 != 0:
            print(f"  [Warning] File has odd number of rows ({len(df)}). "
                  f"Adjacency assumption might be broken for the last row in {os.path.basename(file_path)}.")
        
        # 3. 核心修复逻辑：基于行号生成唯一后缀
        # 逻辑：每一对 (Row i, Row i+1) 将共享同一个 unique_suffix
        # np.arange(len(df)) // 2 生成序列: 0, 0, 1, 1, 2, 2 ...
        pair_indices = np.arange(len(df)) // 2
        
        # 将原始 pair_id 转换为字符串以确保兼容
        df['pair_id'] = df['pair_id'].astype(str)
        
        # 创建新的 pair_id: "原始ID_行对索引"
        # 例如: "1147" -> "1147_0", "1147" -> "1147_1"
        df['pair_id'] = df['pair_id'] + "_" + pd.Series(pair_indices).astype(str)
        
        # 4. (可选) 验证逻辑：检查相邻两行的 pair_id 是否确实相同
        # 这一步是为了防止文件已经被打乱顺序
        # 但既然是刚刚生成的，我们默认顺序是保留的
        
        # 5. 原地保存 (覆盖原文件)
        # 如果你担心安全，可以将 out_path 改为 file_path.replace(".parquet", "_fixed.parquet")
        df.to_parquet(file_path, index=False)
        print(f"  [Fixed] Updated {len(df)} rows in {os.path.basename(file_path)}")
        
    except Exception as e:
        print(f"  [Error] Failed to process {file_path}: {e}")

def main():
    # 指定需要修复的目录
    # 这里涵盖了 recognition 和 selection 两个任务的结果
    target_dirs = [
        "data/results/predictions/recognition",
        "data/results/predictions/selection",
        # 如果 continuation 也有这个问题，也可以加进来
        # "data/results/predictions/continuation" 
    ]
    
    print(">>> Starting Pair ID Fix Task...")
    print("    Logic: Appending sequence index to pair_id based on adjacency (Row N, N+1).")
    
    files_found = []
    for d in target_dirs:
        if os.path.exists(d):
            files_found.extend(glob.glob(os.path.join(d, "*.parquet")))
    
    if not files_found:
        print("No parquet files found.")
        return

    print(f">>> Found {len(files_found)} files to process.")
    
    for file_path in tqdm(files_found, desc="Fixing Files"):
        fix_parquet_pair_ids(file_path)
        
    print(">>> All files processed. You can now run run_eval.py directly.")

if __name__ == "__main__":
    main()