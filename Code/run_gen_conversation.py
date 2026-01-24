import argparse
import os
import glob
import json
import time
import pandas as pd
from tqdm import tqdm
from src.models import load_model
from configs import get_dataset_config 

def main(args):
    print(f">>> Loading config for dataset: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)

    # 1. 扫描文件
    target_files = []
    data_dir = getattr(dataset_config, "data_dir", "data/processed/conversation/")
    print(f">>> Scanning files in {data_dir}...")
    for attr in args.attributes:
        pattern = os.path.join(data_dir, f"{attr}_*.jsonl")
        target_files.extend(glob.glob(pattern))
    
    if not target_files:
        print("Error: No matching files found.")
        return
    
    target_files = sorted(list(set(target_files)))
    print(f">>> Found {len(target_files)} files.")

    # 2. 构建 Prompts
    all_samples = []
    for file_path in target_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                prompt = dataset_config.gen_prompt_template.format(text=record.get("text", ""))
                meta = record.copy()
                meta["_prompt_for_inf"] = prompt
                all_samples.append(meta)

    print(f">>> Total samples to generate: {len(all_samples)}")

    # 3. 加载模型
    model_kwargs = {}
    if args.api_key: model_kwargs['api_key'] = args.api_key
    if args.base_url: model_kwargs['base_url'] = args.base_url
    if args.max_workers: model_kwargs['max_workers'] = args.max_workers
    
    model = load_model(
        model_type=args.model_type,
        model_path_or_name=args.model_path_or_name if hasattr(args, 'model_path_or_name') else args.model_path,
        **model_kwargs
    )

    # 4. 批量生成 (带进度条)
    prompts = [s["_prompt_for_inf"] for s in all_samples]
    if args.debug_n > 0:
        prompts = prompts[:args.debug_n]
        all_samples = all_samples[:args.debug_n]

    results = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Response", unit="batch"):
        batch_prompts = prompts[i : i + batch_size]
        batch_output = model.predict_batch(batch_prompts, max_new_tokens=args.max_new_tokens)
        
        for j, text_out in enumerate(batch_output):
            sample_idx = i + j
            meta = all_samples[sample_idx]
            results.append({
                **meta,
                "raw_response": text_out,
                "model_name": args.model_name
            })
            if "_prompt_for_inf" in results[-1]:
                del results[-1]["_prompt_for_inf"]

    # 5. 保存结果
    df = pd.DataFrame(results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/results/predictions/conversation", exist_ok=True)
    out_path = f"data/results/predictions/conversation/{args.model_name}_{timestamp}.parquet"
    
    df.to_parquet(out_path, index=False)
    print(f">>> Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="holistic_conversation")
    parser.add_argument("--attributes", nargs='+', required=True)
    parser.add_argument("--model_type", type=str, default="hf")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--debug_n", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    args = parser.parse_args()
    main(args)