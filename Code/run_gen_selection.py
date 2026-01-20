# run_gen_selection.py
import argparse
import os
import glob
import json
import time
import pandas as pd
from tqdm import tqdm
from src.models import load_model
from configs import get_dataset_config 

def parse_response(response: str, label_map: dict) -> int:
    """解析 MCQ 输出 (0, 1, 2)"""
    response = response.strip()
    if not response: return -1
    
    # 优先精确匹配
    if response in label_map:
        return label_map[response]
    
    # 模糊匹配 (处理 "Option 0", "Answer: 1" 等)
    # 这种简单逻辑通常足够，如果模型啰嗦，可以加正则
    for key in ["0", "1", "2"]:
        if key in response:
            return label_map[key]
    return -1

def main(args):
    # 1. 加载配置
    print(f">>> Loading config for dataset: {args.dataset}")
    try:
        dataset_config = get_dataset_config(args.dataset)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # 2. 扫描文件
    target_files = []
    data_dir = getattr(dataset_config, "data_dir", "data/processed/selection/")
    print(f">>> Scanning files for attributes: {args.attributes}")
    for attr in args.attributes:
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
    
    target_files = sorted(list(set(target_files)))
    if not target_files:
        print("Error: No matching files found.")
        return
    
    print(f">>> Found {len(target_files)} files.")

    # 3. 构建 Prompts 并提取元数据
    all_samples = []
    for file_path in target_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                
                prompt = dataset_config.prompt_template.format(
                    context=record.get("context", ""),
                    question=record.get("question", ""),
                    ans0=record.get("ans0", ""),
                    ans1=record.get("ans1", ""),
                    ans2=record.get("ans2", "")
                )
                
                compact_record = {
                    "sample_id": record["sample_id"],
                    "pair_id": record["pair_id"],
                    "pair_type": record["pair_type"],
                    "base_attr": record["base_attr"],
                    "add_attr": record["add_attr"],
                    "add_attr_value": record["add_attr_value"],
                    "label": record["label"],
                    # [新增] 关键字段：区分模棱两可还是明确
                    "context_condition": record.get("context_condition", "unknown"), 
                    "_prompt_for_inf": prompt
                }
                all_samples.append(compact_record)

    print(f">>> Total samples: {len(all_samples)}")

    # 4. 加载模型
    model_kwargs = {}
    if args.api_key: model_kwargs['api_key'] = args.api_key
    if args.base_url: model_kwargs['base_url'] = args.base_url
    if args.max_workers: model_kwargs['max_workers'] = args.max_workers
    
    model = load_model(
        model_type=args.model_type,
        model_path_or_name=args.model_path_or_name if hasattr(args, 'model_path_or_name') else args.model_path,
        **model_kwargs
    )

    # 5. 批量推理
    prompts = [s["_prompt_for_inf"] for s in all_samples]
    if args.debug_n > 0:
        prompts = prompts[:args.debug_n]
        all_samples = all_samples[:args.debug_n]

    results = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(prompts), batch_size), desc="Inferencing"):
        batch_prompts = prompts[i : i + batch_size]
        batch_output = model.predict_batch_with_prob(batch_prompts)
        
        for j, out in enumerate(batch_output):
            sample_idx = i + j
            meta = all_samples[sample_idx]
            
            pred_label = parse_response(out['text'], dataset_config.label_map)
            
            results.append({
                **meta,
                "pred_label": pred_label,
                # "pred_prob": out['probability'],
                "raw_response": out['text'],
                "model_name": args.model_name
            })
            if "_prompt_for_inf" in results[-1]:
                del results[-1]["_prompt_for_inf"]

    # 6. 保存结果
    df = pd.DataFrame(results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/results/predictions/selection", exist_ok=True)
    out_path = f"data/results/predictions/selection/{args.model_name}_{timestamp}.parquet"
    
    df.to_parquet(out_path, index=False)
    print(f">>> Saved predictions to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bbq_selection")
    parser.add_argument("--attributes", nargs='+', required=True)
    parser.add_argument("--model_type", type=str, default="hf")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--debug_n", type=int, default=0)
    
    args = parser.parse_args()
    main(args)