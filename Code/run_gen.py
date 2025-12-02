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