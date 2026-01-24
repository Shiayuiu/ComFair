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
    # 1. 加载配置
    print(f">>> Loading config for dataset: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)

    # 2. 扫描文件
    target_files = []
    data_dir = getattr(dataset_config, "data_dir", "data/processed/continuation/")
    
    # 逻辑: 扫描所有匹配 attributes 的文件
    # 文件名范式: [base]_[add]_[value].jsonl
    print(f">>> Scanning files in {data_dir}...")
    for attr in args.attributes:
        pattern = os.path.join(data_dir, f"{attr}_*.jsonl")
        found = glob.glob(pattern)
        target_files.extend(found)
    
    target_files = sorted(list(set(target_files)))
    if not target_files:
        print("Error: No matching files found.")
        return
    print(f">>> Found {len(target_files)} files.")

    # 3. 构建 Prompts
    all_samples = []
    for file_path in target_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                
                # 构建生成 Prompt
                prompt = dataset_config.gen_prompt_template.format(
                    text=record.get("text", "")
                )
                
                # 保留所有元数据以便 eval
                meta = record.copy()
                meta["_prompt_for_inf"] = prompt
                all_samples.append(meta)

    print(f">>> Total samples to generate: {len(all_samples)}")

    # 4. 加载目标模型 (待测模型)
    model_kwargs = {}
    if args.api_key: model_kwargs['api_key'] = args.api_key
    if args.base_url: model_kwargs['base_url'] = args.base_url
    if args.max_workers: model_kwargs['max_workers'] = args.max_workers
    
    model = load_model(
        model_type=args.model_type,
        model_path_or_name=args.model_path_or_name if hasattr(args, 'model_path_or_name') else args.model_path,
        **model_kwargs
    )

    # 5. 批量推理 (续写)
    prompts = [s["_prompt_for_inf"] for s in all_samples]
    if args.debug_n > 0:
        prompts = prompts[:args.debug_n]
        all_samples = all_samples[:args.debug_n]

    results = []
    batch_size = args.batch_size

    # Continuation 任务通常输出较长，注意 max_new_tokens
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", unit="batch"):
        batch_prompts = prompts[i : i + batch_size]
        # 不需要概率，只需要生成的文本
        batch_output = model.predict_batch(batch_prompts, max_new_tokens=args.max_new_tokens)
        
        for j, text_out in enumerate(batch_output):
            sample_idx = i + j
            meta = all_samples[sample_idx]
            
            results.append({
                **meta,
                "raw_response": text_out, # 模型的续写内容
                "model_name": args.model_name
            })
            if "_prompt_for_inf" in results[-1]:
                del results[-1]["_prompt_for_inf"]

    # 6. 保存
    df = pd.DataFrame(results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/results/predictions/continuation", exist_ok=True)
    out_path = f"data/results/predictions/continuation/{args.model_name}_{timestamp}.parquet"
    
    df.to_parquet(out_path, index=False)
    print(f">>> Generation saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="holistic_continuation")
    parser.add_argument("--attributes", nargs='+', required=True)
    parser.add_argument("--model_type", type=str, default="hf")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--debug_n", type=int, default=0)
    # Continuation 任务特有参数
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Length of continuation")
    
    args = parser.parse_args()
    main(args)