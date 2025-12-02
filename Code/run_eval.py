#!/usr/bin/env python3
"""
LLM交叉公平性评测 - 评估模块
对预测结果进行交叉公平性评估
"""

import os
import argparse
import pandas as pd
import json
import sys
from pathlib import Path

# 添加src到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from metrics.intersectional import evaluate_intersectional_fairness
from metrics.group_validation import validate_intersectional_groups
from utils.reporting import generate_fairness_report
from configs import get_dataset_config


def parse_args():
    parser = argparse.ArgumentParser(description="评估LLM预测结果的交叉公平性")
    parser.add_argument('--prediction-file', type=str, required=True,
                       help='预测结果文件路径')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集名称')
    parser.add_argument('--output-dir', type=str, 
                       default='data/results/reports',
                       help='评估报告输出目录')
    parser.add_argument('--dimensions', type=str, nargs='+', 
                       default=['all'],
                       help='交叉评估维度，如 "gender" "gender race" 或 "all"')
    parser.add_argument('--min-group-size', type=int, default=10,
                       help='最小群体样本量')
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                       help='要计算的指标')
    parser.add_argument('--include-baseline', action='store_true',
                       help='包含整体基准指标')
    return parser.parse_args()


def load_predictions(file_path):
    """加载预测结果"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"预测文件不存在: {file_path}")
    
    return pd.read_parquet(file_path)


def determine_evaluation_dimensions(dimensions_arg, dataset_config):
    """确定要评估的交叉维度"""
    if dimensions_arg == ['all']:
        # 使用配置中的所有交叉维度
        return dataset_config.intersectional_dims
    
    evaluation_dims = []
    for dim_str in dimensions_arg:
        dims = dim_str.split()
        if all(dim in dataset_config.sensitive_attributes for dim in dims):
            evaluation_dims.append(dims)
        else:
            print(f"警告: 维度 {dims} 不在数据集敏感属性中，已跳过")
    
    return evaluation_dims


def main():
    args = parse_args()
    
    # 加载数据集配置
    print(f"加载数据集配置: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)
    
    # 加载预测结果
    print(f"加载预测结果: {args.prediction_file}")
    predictions_df = load_predictions(args.prediction_file)
    
    # 确定评估维度
    evaluation_dims = determine_evaluation_dimensions(
        args.dimensions, dataset_config
    )
    
    if not evaluation_dims:
        raise ValueError("没有有效的评估维度")
    
    print(f"评估维度: {evaluation_dims}")
    
    # 验证数据集完整性
    required_columns = ['sample_id', 'y_pred'] + list(dataset_config.sensitive_attributes.keys())
    missing_columns = [col for col in required_columns if col not in predictions_df.columns]
    if missing_columns:
        raise ValueError(f"预测结果缺少必要列: {missing_columns}")
    
    # 执行交叉公平性评估
    print("开始交叉公平性评估...")
    fairness_results = {}
    
    for dims in evaluation_dims:
        print(f"评估交叉维度: {dims}")
        
        # 群体有效性验证
        valid_groups = validate_intersectional_groups(
            predictions_df, 
            dims, 
            min_size=args.min_group_size
        )
        
        if not valid_groups:
            print(f"警告: 维度 {dims} 下无有效群体，跳过")
            continue
        
        # 计算交叉公平性指标
        dim_results = evaluate_intersectional_fairness(
            predictions_df=predictions_df,
            sensitive_attributes=dims,
            metrics=args.metrics,
            valid_groups=valid_groups,
            include_baseline=args.include_baseline
        )
        
        dimension_key = '_'.join(dims)
        fairness_results[dimension_key] = dim_results
    
    if not fairness_results:
        raise RuntimeError("所有评估维度均无有效结果")
    
    # 生成评估报告
    print("生成评估报告...")
    report = generate_fairness_report(
        fairness_results=fairness_results,
        dataset_name=args.dataset,
        model_info={
            'model': predictions_df['model'].iloc[0] if 'model' in predictions_df.columns else 'unknown',
            'model_name': predictions_df['model_name'].iloc[0] if 'model_name' in predictions_df.columns else 'unknown'
        },
        evaluation_config={
            'dimensions_evaluated': evaluation_dims,
            'min_group_size': args.min_group_size,
            'metrics_calculated': args.metrics
        }
    )
    
    # 保存报告
    os.makedirs(args.output_dir, exist_ok=True)
    prediction_filename = Path(args.prediction_file).stem
    report_filename = f"{prediction_filename}_fairness_report.json"
    report_path = os.path.join(args.output_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"评估报告已保存: {report_path}")
    
    # 打印摘要
    print("\n" + "="*50)
    print("交叉公平性评估摘要")
    print("="*50)
    for dim_key, results in fairness_results.items():
        print(f"\n维度: {dim_key}")
        if 'fairness_metrics' in results:
            fairness_metrics = results['fairness_metrics']
            print(f"  最大差异: {fairness_metrics.get('max_disparity', 'N/A'):.4f}")
            print(f"  标准差: {fairness_metrics.get('std_disparity', 'N/A'):.4f}")
            print(f"  最差群体表现: {fairness_metrics.get('min_group_performance', 'N/A'):.4f}")
    
    return report_path


if __name__ == "__main__":
    main()