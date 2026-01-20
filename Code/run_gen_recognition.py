# run_gen_recognition.py
import argparse
import os
import glob
import json
import time
import pandas as pd
from tqdm import tqdm
from src.models import load_model
from configs import get_dataset_config 

# 辅助函数：解析标签
def parse_response(response: str, label_map: dict) -> int:
    """
    尝试从模型输出中提取 1 或 2，并映射到内部 label (0/1)
    """
    response = response.strip()
    # 简单规则：查找是否包含 1 或 2
    if "1" in response:
        return label_map.get("1", -1)
    elif "2" in response:
        return label_map.get("2", -1)
    return -1

def main(args):
    # 1. 加载配置
    print(f">>> Loading config for dataset: {args.dataset}")
    try:
        # 这里假设你已经在 configs/__init__.py 里注册了 bbq_recognition
        dataset_config = get_dataset_config(args.dataset)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # 2. 扫描数据文件
    # 逻辑：对于传入的每个 attr (如 race)，寻找 data_dir 下所有以 "race_" 开头的文件
    target_files = []
    data_dir = getattr(dataset_config, "data_dir", "data/processed/recognition/")
    
    print(f">>> Scanning files for attributes: {args.attributes}")
    for attr in args.attributes:
        # 查找 pattern: [base_attr]_[add_attr].jsonl
        # 例如 base=race -> data/processed/recognition/race_*.jsonl
        pattern = os.path.join(data_dir, f"{attr}_*.jsonl")
        found = glob.glob(pattern)
        # 仅保留另一维属性也在 args.attributes 列表中的文件
        for fpath in found:
            fname = os.path.basename(fpath)
            name, _ext = os.path.splitext(fname)
            parts = name.split("_")
            if len(parts) >= 2:
                other = parts[1]
                if other in args.attributes:
                    target_files.append(fpath)
    
    target_files = sorted(list(set(target_files))) # 去重
    if not target_files:
        print("Error: No matching files found.")
        return
    
    print(f">>> Found {len(target_files)} files: {[os.path.basename(f) for f in target_files]}")

    # 3. 加载并构建 Prompt
    all_samples = []
    print(">>> Preparing prompts...")
    
    for file_path in target_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                
                # 构建 Prompt
                prompt = dataset_config.prompt_template.format(
                    context=record[dataset_config.column_mapping['context']],
                    text=record[dataset_config.column_mapping['text']]
                )
                
                # 仅保留关键元数据，丢弃 text/context/prompt/sample_size
                compact_record = {
                    "sample_id": record["sample_id"],
                    "pair_id": record["pair_id"],
                    "pair_type": record["pair_type"],         # 'base' or 'augmented'
                    "base_attr": record["base_attr"],         # e.g. 'race'
                    "add_attr": record["add_attr"],           # e.g. 'gender'
                    "add_attr_value": record["add_attr_value"], # e.g. 'original' or 'female'
                    "label": record["label"],
                    "_prompt_for_inf": prompt # 临时字段，推理后删除
                }
                all_samples.append(compact_record)

    print(f">>> Total samples to inference: {len(all_samples)}")

    # 4. 加载模型
    model_kwargs = {}
    if args.api_key: model_kwargs['api_key'] = args.api_key
    if args.base_url: model_kwargs['base_url'] = args.base_url
    if args.max_workers: model_kwargs['max_workers'] = args.max_workers
    
    model = load_model(
        model_type=args.model_type,
        model_path_or_name=args.model_path,
        **model_kwargs
    )

    # 5. 批量推理
    prompts = [s["_prompt_for_inf"] for s in all_samples]
    batch_size = args.batch_size
    
    results = []
    
    # 调试模式
    if args.debug_n > 0:
        print(f"!!! DEBUG MODE: Processing only first {args.debug_n} samples !!!")
        prompts = prompts[:args.debug_n]
        all_samples = all_samples[:args.debug_n]

    for i in tqdm(range(0, len(prompts), batch_size), desc="Inferencing"):
        batch_prompts = prompts[i : i + batch_size]
        batch_output = model.predict_batch_with_prob(batch_prompts)
        
        for j, out in enumerate(batch_output):
            sample_idx = i + j
            meta = all_samples[sample_idx]
            
            # 解析结果
            pred_label = parse_response(out['text'], dataset_config.label_map)
            
            # 组装最终行 (移除临时 Prompt)
            result_row = {
                "sample_id": meta["sample_id"],
                "pair_id": meta["pair_id"],
                "pair_type": meta["pair_type"],
                "base_attr": meta["base_attr"],
                "add_attr": meta["add_attr"],
                "add_attr_value": meta["add_attr_value"],
                "label": meta["label"],
                "pred_label": pred_label,
                # "pred_prob": out['probability'],
                "raw_response": out['text'], # 可选保留用于 debug
                "model_name": args.model_name
            }
            results.append(result_row)

    # 6. 保存结果
    df = pd.DataFrame(results)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/results/predictions/recognition", exist_ok=True)
    out_path = f"data/results/predictions/recognition/{args.model_name}_{timestamp}.parquet"
    
    df.to_parquet(out_path, index=False)
    print(f">>> Saved {len(df)} predictions to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bbq_augmented")
    parser.add_argument("--attributes", nargs='+', required=True, help="List of base attributes to process, e.g., race gender")
    parser.add_argument("--model_type", type=str, default="hf", choices=["hf", "api", "reasoning_api"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--debug_n", type=int, default=0)
    
    args = parser.parse_args()
    main(args)