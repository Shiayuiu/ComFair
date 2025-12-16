# run_gen.py
import re
import argparse
import pandas as pd
import os
import time
#from configs.adult import dataset_config
from configs import get_dataset_config # Dynamic Import
from src.models.huggingface import HFModelWrapper

def parse_response_backup(response_text: str, label_map: dict) -> int:
    """
    简单解析器，将模型输出映射为 0 或 1。
    如果无法解析，返回 -1
    """
    # 简单清洗：去掉标点
    clean_text = response_text.strip().upper().replace(".", "")
    
    # 优先精确匹配
    if clean_text in label_map:
        return label_map[clean_text]
    
    # 模糊匹配 (处理 "Option A", "Result: A" 等情况)
    for key, val in label_map.items():
        if key in clean_text:
            return val
            
    return -1 # 解析失败

def parse_response(response_text: str, label_map: dict) -> int:
    """
    鲁棒的解析器：优先匹配开头的单个字母，防止匹配到解释文本中的字符。
    """
    if not response_text:
        return -1
    
    text = response_text.strip().upper()

    # 策略 1 (最优先)：直接提取第一个“单词”进行匹配
    # 能够解决 "B\n\nExplanation..." 或 "A." 或 "A," 的情况
    # 按空白字符切分，取第一个
    first_token = text.split()[0]
    
    # 去除单词末尾的标点 (例如 "A." -> "A", "B," -> "B")
    clean_token = re.sub(r"[^\w]", "", first_token)
    
    if clean_token in label_map:
        return label_map[clean_token]

    # 策略 2 (备选)：如果开头是 "Answer: A" 这种格式
    # 使用正则寻找独立的 "A" 或 "B"
    for key, val in label_map.items():
        # 正则逻辑：
        # (?:^|\s)  : 前面必须是 开头 或者 空格/换行
        # key       : 匹配 A 或 B
        # (?:\.|,|$|\s): 后面必须是 标点、结尾 或者 空格
        # 这样可以防止匹配到 "Apple" 中的 "A" 或 "Label" 中的 "B"
        pattern = fr"(?:^|\s|:){key}(?:\.|,|$|\s)"
        if re.search(pattern, text):
            return val
            
    return -1 # 解析失败

def main(args):
    
    # 1. 准备配置和路径
    print(f"Loading configuration for dataset: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset) # 关键修改

    output_dir = "data/results/predictions"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(output_dir, f"{args.model_name}-{args.dataset}-{timestamp}.parquet")
    
    # 2. 加载数据 (只加载 feature csv)
    print(f"Loading features from {dataset_config.processed_feature_path}...")
    df = pd.read_csv(dataset_config.processed_feature_path)
    
    # 调试模式：只跑前 N 条
    if args.debug_n > 0:
        df = df.head(args.debug_n)
    
    # 3. 构建 Prompts
    # 利用 DataFrame 的行与 Config 中的 map 动态构建
    print("Constructing prompts...")
    prompts = []
    
    for _, row in df.iterrows():
        # 动态将 standard_key 转换为 row 中的实际列名
        format_dict = {
            std_key: row[csv_col] 
            for std_key, csv_col in dataset_config.column_mapping.items()
        }
        prompt = dataset_config.prompt_template.format(**format_dict)
        prompts.append(prompt)
    
    # 4. 加载模型
    model = HFModelWrapper(model_path=args.model_path)
    
    # 5. 批量推理
    print(f"Running inference on {len(prompts)} samples...")
    batch_size = args.batch_size
    all_preds = []
    all_raw_texts = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_responses = model.predict_batch(batch_prompts)
        
        for resp in batch_responses:
            all_raw_texts.append(resp)
            all_preds.append(parse_response(resp, dataset_config.label_map))
            
        if i % 100 == 0:
            print(f"Processed {i}/{len(prompts)}")

    # 6. 结果整合与保存
    result_df = pd.DataFrame({
        "sample_id": df["sample_id"],   # 关键：用于 Join
        # "prompt": prompts,              # 可选：便于调试
        "raw_response": all_raw_texts,
        "pred_label": all_preds,
        "model_name": args.model_name
    })
    
    # 过滤解析失败的 (-1) 可以在这里做，也可以在 Eval 阶段做
    # result_df = result_df[result_df['pred_label'] != -1]

    result_df.to_parquet(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-1.5B", help="Model name identifier")
    parser.add_argument("--model_path", type=str, required=True, help="Local path or HF hub path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--debug_n", type=int, default=0, help="Run only N samples for debugging")
    
    args = parser.parse_args()
    main(args)