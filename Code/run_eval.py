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
from configs import get_dataset_config # 动态加载配置

class NpEncoder(json.JSONEncoder):
    """解决 NumPy 数据类型无法 JSON 序列化的问题"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main(args):
    start_time = time.time()
    
    # 1. 动态加载配置
    print(f">>> Loading configuration for dataset: {args.dataset}")
    try:
        dataset_config = get_dataset_config(args.dataset)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # 2. 路径与数据加载
    print(f">>> Loading predictions from: {args.pred_path}")
    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(f"Prediction file not found: {args.pred_path}")
        
    pred_df = pd.read_parquet(args.pred_path)
    
    print(f">>> Loading metadata from: {dataset_config.processed_metadata_path}")
    meta_df = pd.read_csv(dataset_config.processed_metadata_path)
    
    # 3. 数据合并 (Inner Join on sample_id)
    # 这一步将模型的预测结果与它从未见过的敏感属性合并
    full_df = pd.merge(pred_df, meta_df, on="sample_id", how="inner")
    print(f"Merged Data Shape: {full_df.shape}")
    
    # 检查是否有概率列 (pred_prob)
    has_prob = 'pred_prob' in full_df.columns
    if has_prob:
        print("Detail: Probability column found. Calibration & AUC metrics enabled.")
    else:
        print("Warning: No probability column found. Skipping Calibration & AUC.")

    # 4. 全局性能评估 (Overall Performance)
    # 过滤掉无法解析的结果 (-1)
    valid_mask = full_df['pred_label'] != -1
    valid_df = full_df[valid_mask].copy()
    
    dropped_count = len(full_df) - len(valid_df)
    if dropped_count > 0:
        print(f"Warning: Dropped {dropped_count} unparseable samples.")
    
    overall_metrics = compute_basic_metrics(valid_df['label'], valid_df['pred_label'])
    
    # 5. 交叉公平性评估初始化
    # 注意：这里传入 prob_col 以启用高级指标
    evaluator = IntersectionalEvaluator(
        valid_df, 
        label_col='label', 
        pred_col='pred_label',
        prob_col='pred_prob' if has_prob else None
    )
    
    fairness_report = {}
    
    # 6. 定义扫描范围
    # 获取敏感属性列表，如果没有定义则使用默认
    sensitive_attrs = getattr(dataset_config, 'sensitive_attributes', ['race', 'gender'])
    
    scan_list = []
    # Level 1: 单属性 (e.g., Race)
    scan_list.extend([[attr] for attr in sensitive_attrs])
    # Level 2: 双属性交叉 (e.g., Race + Gender)
    scan_list.extend([list(c) for c in combinations(sensitive_attrs, 2)])
    
    print(f">>> Starting intersectional scan on {len(scan_list)} combinations...")
    
    # 7. 执行扫描
    for group_keys in scan_list:
        group_name = " & ".join(group_keys)
        # print(f"Evaluating: [{group_name}]") # 可选：减少刷屏
        
        # 构建该组合的所有指标
        metrics_bundle = {
            # --- 基础差异 (Disparity) ---
            "accuracy_parity": evaluator.compute_subgroup_performance(group_keys, metric='accuracy'),
            "equalized_odds": evaluator.evaluate_equal_opportunity(group_keys), # Recall parity
            "predictive_parity": evaluator.evaluate_predictive_parity(group_keys), # Precision parity
            
            # --- 高级公平性 (The "Fancy" Metrics) ---
            "disparate_impact": evaluator.evaluate_disparate_impact(group_keys),
            "differential_fairness": evaluator.evaluate_differential_fairness(group_keys),
            "treatment_equality": evaluator.evaluate_treatment_equality(group_keys),
            "distributional_fairness": evaluator.evaluate_distributional_fairness(group_keys, metric='accuracy')
        }
        
        # --- 校准与置信度 (仅当有概率时) ---
        if has_prob:
            metrics_bundle["calibration_fairness"] = evaluator.evaluate_calibration_fairness(group_keys)
            metrics_bundle["auc_parity"] = evaluator.compute_subgroup_performance(group_keys, metric='auc')
            
        fairness_report[group_name] = metrics_bundle

    # 8. 生成最终报告结构
    final_report = {
        "meta_info": {
            "model": args.model_name,
            "dataset": args.dataset,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(full_df),
            "valid_samples": len(valid_df),
            "sensitive_attributes_scanned": sensitive_attrs,
            "has_probability": has_prob
        },
        "overall_performance": overall_metrics,
        "fairness_analysis": fairness_report
    }
    
    # 9. 保存报告
    os.makedirs("data/results/reports", exist_ok=True)
    report_filename = os.path.basename(args.pred_path).replace(".parquet", "_report.json")
    report_path = os.path.join("data/results/reports", report_filename)
    
    with open(report_path, "w") as f:
        # 使用自定义 Encoder 处理 NumPy 数据类型
        json.dump(final_report, f, indent=4, cls=NpEncoder)
        
    elapsed = time.time() - start_time
    print(f">>> Evaluation Complete in {elapsed:.2f}s.")
    print(f">>> Report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True, help="Path to prediction parquet file")
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name for config loading")
    parser.add_argument("--model_name", type=str, default="Unknown", help="Model name for reporting")
    
    args = parser.parse_args()
    main(args)