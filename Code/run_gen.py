<<<<<<< HEAD
#!/usr/bin/env python3
"""
LLM交叉公平性评测 - 预测生成模块
生成模型预测结果并保存到缓存文件
"""

import os
import argparse
import pandas as pd
from datetime import datetime
import sys

# 添加src到路径
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.loader import load_dataset
from src.models.llm_client import LLMClient
from src.utils.validation import validate_predictions
from configs import get_dataset_config


def parse_args():
    parser = argparse.ArgumentParser(description="生成LLM预测结果")
    parser.add_argument('--dataset', type=str, required=True, 
                       help='数据集名称')
    parser.add_argument('--model', type=str, required=True,
                       choices=['openai', 'anthropic', 'azure', 'huggingface'],
                       help='LLM后端类型')
    parser.add_argument('--model-name', type=str, default=None,
                       help='具体模型名称，如 gpt-4, claude-3等')
    parser.add_argument('--output-dir', type=str, 
                       default='data/results/predictions',
                       help='预测结果输出目录')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大样本数（用于调试）')
    parser.add_argument('--cache-only', action='store_true',
                       help='仅使用缓存，不调用API')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载数据集配置
    print(f"加载数据集配置: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)
    
    # 加载数据
    print("加载数据集...")
    features_df, metadata_df = load_dataset(args.dataset)
    
    if args.max_samples:
        features_df = features_df.head(args.max_samples)
        metadata_df = metadata_df.head(args.max_samples)
    
    # 初始化LLM客户端
    print(f"初始化LLM客户端: {args.model}")
    llm_client = LLMClient(
        backend=args.model,
        model_name=args.model_name,
        cache_dir=args.output_dir
    )
    
    # 生成预测
    print("开始生成预测...")
    predictions = []
    
    for idx, (_, feature_row) in enumerate(features_df.iterrows()):
        print(f"处理样本 {idx + 1}/{len(features_df)}")
        
        # 获取对应的元数据
        sample_id = feature_row['sample_id']
        meta_row = metadata_df[metadata_df['sample_id'] == sample_id].iloc[0]
        
        # 构建prompt - 注入敏感属性
        prompt = dataset_config.build_prompt(
            text=feature_row['text'],
            sensitive_attributes={attr: meta_row[attr] for attr in dataset_config.sensitive_attributes}
        )
        
        # 获取预测
        if args.cache_only:
            prediction = llm_client.get_cached_prediction(prompt)
        else:
            prediction = llm_client.get_prediction(prompt)
        
        if prediction:
            # 构建结果记录
            result = {
                'sample_id': sample_id,
                'text': feature_row['text'],
                'prompt_used': prompt,
                'y_pred': prediction['label'],
                'y_score': prediction.get('confidence', 0.5),
                'model': args.model,
                'model_name': args.model_name or llm_client.default_model_name,
                'dataset': args.dataset,
                'timestamp': datetime.now().isoformat()
            }
            
            # 添加敏感属性
            for attr in dataset_config.sensitive_attributes:
                result[attr] = meta_row[attr]
                
            # 添加真实标签（如果存在）
            if 'label' in feature_row:
                result['y_true'] = feature_row['label']
                
            predictions.append(result)
    
    # 转换为DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # 验证预测结果
    print("验证预测结果...")
    validation_result = validate_predictions(predictions_df, dataset_config)
    if not validation_result['valid']:
        print(f"警告: 预测结果验证失败: {validation_result['errors']}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.model}-{args.dataset}-{timestamp}.parquet"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, filename)
    
    predictions_df.to_parquet(output_path)
    print(f"预测结果已保存: {output_path}")
    print(f"总样本数: {len(predictions_df)}")
    
    # 保存元信息
    meta_info = {
        'dataset': args.dataset,
        'model': args.model,
        'model_name': args.model_name,
        'timestamp': timestamp,
        'total_samples': len(predictions_df),
        'sensitive_attributes': list(dataset_config.sensitive_attributes.keys())
    }
    
    meta_filename = f"{args.model}-{args.dataset}-{timestamp}_meta.json"
    pd.Series(meta_info).to_json(os.path.join(args.output_dir, meta_filename))
    
    return output_path


if __name__ == "__main__":
    main()
=======
# run_gen.py
import re
import argparse
import pandas as pd
import os
import time
from configs import get_dataset_config  # Dynamic Import
from src.models import load_model

### [MOD] 1) 新增：tqdm 进度条
from tqdm import tqdm

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

    return -1  # 解析失败

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

    return -1  # 解析失败

def main(args):

    # 1. 准备配置和路径
    print(f"Loading configuration for dataset: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)  # 关键修改
    output_dir = "data/results/predictions/classification/"
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
    # 构造参数字典
    model_kwargs = {}
    if args.api_key:
        model_kwargs['api_key'] = args.api_key
    if args.base_url:
        model_kwargs['base_url'] = args.base_url

    # 统一加载
    model = load_model(
        model_type=args.model_type,
        model_path_or_name=args.model_path,  # 对于 API，这里传入 "gpt-4" 等名称
        **model_kwargs
    )

    # 5. 批量推理
    print(f"Running inference on {len(prompts)} samples...")
    batch_size = args.batch_size

    all_raw_texts = []
    all_preds = []
    all_probs = []  # <--- 新增：存储概率

    ### [MOD] 2) 替换：用 tqdm 做“按样本数”的进度条（不会再只有 0 和跑满）
    # 原逻辑是 i%100 打印，i 是 batch 起始索引，很多情况下只会打印 0
    pbar = tqdm(total=len(prompts), desc="Inference", unit="sample")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]

        # 调用新方法
        batch_results = model.predict_batch_with_prob(batch_prompts)

        for res in batch_results:
            text = res['text']
            prob = res['probability']

            # 解析标签 (A->1, B->0)
            pred_label = parse_response(text, dataset_config.label_map)

            all_raw_texts.append(text)
            all_preds.append(pred_label)
            all_probs.append(prob)  # <--- 存储

        ### [MOD] 3) 新增：每完成一个 batch，就按实际返回条数更新进度
        pbar.update(len(batch_results))

    ### [MOD] 4) 新增：结束时关闭进度条
    pbar.close()

    # 6. 结果整合与保存
    result_df = pd.DataFrame({
        "sample_id": df["sample_id"],
        "prompt": prompts,
        "raw_response": all_raw_texts,
        "pred_label": all_preds,
        # "pred_prob": all_probs,
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

    parser.add_argument("--model_type", type=str, default="hf", choices=["hf", "api", "reasoning_api"],
                        help="Model backend type: 'hf' for local HuggingFace, 'api' for OpenAI-compatible APIs, 'reasoning_api' for Reasoning models. ")
    parser.add_argument("--api_key", type=str, default=None, help="API Key for API models")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for API models")

    args = parser.parse_args()
    main(args)
>>>>>>> ac22061 (sync latest local version)
