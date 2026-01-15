# run_eval.py
import argparse
import pandas as pd
import json
import os
import time
import numpy as np
from itertools import combinations
from src.metrics.intersectional import IntersectionalEvaluator
from src.metrics.functional import compute_basic_metrics
from configs import get_dataset_config 

class NpEncoder(json.JSONEncoder):
    """解决 NumPy 数据类型无法 JSON 序列化的问题"""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def apply_eval_transforms(df: pd.DataFrame, config) -> pd.DataFrame:
    """[Process] 应用映射逻辑"""
    df_out = df.copy()
    if hasattr(config, 'eval_transforms'):
        for col, func in config.eval_transforms.items():
            if col in df_out.columns:
                target_col = f"{col}_mapped"
                df_out[target_col] = df_out[col].apply(func)
                print(f"  [Transform] {col} -> {target_col}")
    return df_out

def get_query_combinations(config, user_query_attrs: list) -> list:
    """[Query] 解析属性组合"""
    base_attrs = user_query_attrs if user_query_attrs else getattr(config, 'sensitive_attributes', [])
    final_attrs = []
    transforms = getattr(config, 'eval_transforms', {})
    
    for attr in base_attrs:
        # 如果该属性在转换列表中，自动指向映射后的列
        if attr in transforms:
            final_attrs.append(f"{attr}_mapped")
        else:
            final_attrs.append(attr)
            
    scan_list = []
    scan_list.extend([[attr] for attr in final_attrs])
    scan_list.extend([list(c) for c in combinations(final_attrs, 2)])
    return scan_list

def main(args):
    # 1. 初始化与数据加载
    print(f">>> [Init] Dataset: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)
    
    pred_df = pd.read_parquet(args.pred_path)
    meta_df = pd.read_csv(dataset_config.processed_metadata_path)
    
    full_df = pd.merge(pred_df, meta_df, on="sample_id", how="inner")
    
    # 2. [Process] 动态映射
    eval_df = apply_eval_transforms(full_df, dataset_config)
    has_prob = 'pred_prob' in eval_df.columns
    
    # 3. [Eval] 初始化评估器
    evaluator = IntersectionalEvaluator(
        eval_df, 
        label_col='label', 
        pred_col='pred_label', 
        prob_col='pred_prob' if has_prob else None
    )
    
    # 4. [Query] 确定组合
    query_combinations = get_query_combinations(dataset_config, args.query_attrs)
    
    # 5. [Eval Loop] 执行评估
    fairness_report = {}
    for group_keys in query_combinations:
        group_name = " & ".join(group_keys)
        metrics = {
            "accuracy_parity": evaluator.compute_subgroup_performance(group_keys, 'accuracy'),
            "recall_parity": evaluator.compute_subgroup_performance(group_keys, 'recall'),
            "disparate_impact": evaluator.evaluate_disparate_impact(group_keys),
        }
        if has_prob:
            metrics["auc_parity"] = evaluator.compute_subgroup_performance(group_keys, 'auc')
            metrics["calibration_ece"] = evaluator.evaluate_calibration_fairness(group_keys)
        fairness_report[group_name] = metrics

    # 6. [Report] 路径处理逻辑 (FIXED)
    if args.output_path:
        # 如果用户指定了路径，直接使用
        report_path = args.output_path
    else:
        # 如果未指定，生成默认路径：data/results/reports/[model]-[dataset]-timestamp_report.json
        report_filename = os.path.basename(args.pred_path).replace(".parquet", "_report.json")
        report_path = os.path.join("data/results/reports", report_filename)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(report_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    final_output = {
        "meta": {
            "model": args.model_name,
            "dataset": args.dataset,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "queries": [" & ".join(q) for q in query_combinations]
        },
        "overall_metrics": compute_basic_metrics(eval_df['label'], eval_df['pred_label']),
        "intersectional_analysis": fairness_report
    }
    
    with open(report_path, "w") as f:
        json.dump(final_output, f, indent=4, cls=NpEncoder)
        
    print(f">>> [Report] Successfully saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--model_name", type=str, default="Unknown")
    parser.add_argument("--output_path", type=str, default=None, help="Explicit path to save the report")
    parser.add_argument("--query_attrs", nargs='+', default=None)
    args = parser.parse_args()
    main(args)