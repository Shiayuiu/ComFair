# run_eval.py
import argparse
import pandas as pd
import json
import os
import time
from src.metrics.intersectional import IntersectionalEvaluator
from src.metrics.functional import compute_basic_metrics
from configs import get_dataset_config

def main(args):
    # 1. 数据集和路径定义
    print(f"Loading configuration for dataset: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset) # 关键修改
    
    print(f"Loading predictions from: {args.pred_path}")
    print(f"Loading metadata from: {dataset_config.processed_metadata_path}")
    
    # 2. 数据加载
    pred_df = pd.read_parquet(args.pred_path)
    meta_df = pd.read_csv(dataset_config.processed_metadata_path)
    
    # 3. 数据合并 (Inner Join on sample_id)
    # 这一步将模型的预测结果与它从未见过的敏感属性合并
    full_df = pd.merge(pred_df, meta_df, on="sample_id", how="inner")
    print(f"Merged Data Shape: {full_df.shape}")
    
    # 4. 全局性能评估 (Overall Performance)
    # 过滤掉无法解析的结果 (-1)
    valid_df = full_df[full_df['pred_label'] != -1].copy()
    print(f"Valid Samples (Parsable): {len(valid_df)} / {len(full_df)}")
    
    overall_metrics = compute_basic_metrics(valid_df['label'], valid_df['pred_label'])
    
    # 5. 交叉公平性评估
    evaluator = IntersectionalEvaluator(valid_df, label_col='label', pred_col='pred_label')
    
    fairness_report = {}
    
    # 定义我们要扫描的属性组合
    # 策略：扫描所有单属性 + 所有双属性组合
    # 动态获取 config 中的敏感属性
    sensitive_attrs = dataset_config.sensitive_attributes if hasattr(dataset_config, 'sensitive_attributes') else ['race', 'gender']
    
    # A. 单属性分析
    from itertools import combinations
    scan_list = []
    
    # Level 1: 单属性 (e.g., Race)
    scan_list.extend([[attr] for attr in sensitive_attrs])
    
    # Level 2: 双属性交叉 (e.g., Race + Gender)
    scan_list.extend([list(c) for c in combinations(sensitive_attrs, 2)])
    
    # 开始扫描
    for group_keys in scan_list:
        group_name = " & ".join(group_keys)
        print(f"Evaluating intersectional fairness for: [{group_name}]")
        
        fairness_report[group_name] = {
            "accuracy_parity": evaluator.evaluate_accuracy_parity(group_keys),
            "demographic_parity": evaluator.evaluate_demographic_parity(group_keys),
            "equalized_odds": evaluator.evaluate_equalized_odds(group_keys)
        }

    # 6. 生成最终报告
    final_report = {
        "meta_info": {
            "model": args.model_name,
            "dataset": args.dataset,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(valid_df)
        },
        "overall_performance": overall_metrics,
        "fairness_analysis": fairness_report
    }
    
    # 7. 保存
    os.makedirs("data/results/reports", exist_ok=True)
    report_filename = os.path.basename(args.pred_path).replace(".parquet", "_report.json")
    report_path = os.path.join("data/results/reports", report_filename)
    
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=4)
        
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True, help="Path to prediction parquet file")
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--model_name", type=str, default="Unknown")
    
    args = parser.parse_args()
    main(args)