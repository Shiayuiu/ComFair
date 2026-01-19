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

def generate_markdown_report(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model = data["meta"]["model"]
    dataset = data["meta"]["dataset"]
    analysis = data["intersectional_analysis"]

    markdown = f"# Fairness Evaluation Report: {model} on {dataset}\n\n"
    
    # 1. Overall Performance
    markdown += "## 1. Overall Performance\n"
    for k, v in data["overall_metrics"].items():
        markdown += f"- **{k.capitalize()}**: {v:.4f}\n"
    markdown += "\n"

    # 2. Fairness Analysis Table
    markdown += "## 2. Fairness Analysis Table\n"
    markdown += "| Analysis Level / Condition | DP Gap | EO Gap | Unfairness Score |\n"
    markdown += "| :--- | :--- | :--- | :--- |\n"

    # 遍历原有的交集结果
    for key, metrics in analysis.items():
        dp = metrics["DP_Gap"]["value"]
        eo = metrics["EO_Gap"]["value"]
        unf = metrics["Unfairness_Score"]
        markdown += f"| **{key}** (Base) | {dp:.4f} | {eo:.4f} | **{unf:.4f}** |\n"

        # --- 核心逻辑：导出条件交叉性指标 ---
        # 如果是交叉属性 (含有 '&')，则尝试拆解出 "race & gender: male" 等项
        if "&" in key:
            attr_names = [a.strip() for a in key.split("&")]
            # 提取详细子群数据
            dp_details = metrics["DP_Gap"]["details"]
            eo_details = metrics["EO_Gap"]["details"]

            # 找出该组合中涉及的所有单一属性值 (如 Male, Female, Black, White)
            # 逻辑：对于 'race & gender'，如果固定 'Male'，我们就看不同 'race' 间的差距
            for i, fixed_attr in enumerate(attr_names):
                # 确定要对比的另一个属性索引
                target_attr_idx = 1 - i 
                
                # 找出 fixed_attr 维度的所有可能取值 (从 subgroup 字符串中解析)
                # 例如从 "Black & Male" 中提取 "Male"
                unique_vals = set([s.split("&")[i].strip() for s in dp_details.keys()])

                for val in unique_vals:
                    # 过滤出符合当前条件 (如包含 "Male") 的子群
                    sub_pprs = {k: v for k, v in dp_details.items() if k.split("&")[i].strip() == val}
                    sub_eos = {k: v for k, v in eo_details.items() if k.split("&")[i].strip() == val}

                    if len(sub_pprs) > 1:
                        # 计算条件 DP Gap
                        c_dp_gap = max(sub_pprs.values()) - min(sub_pprs.values())
                        
                        # 计算条件 EO Gap
                        tprs = [v["tpr"] for v in sub_eos.values()]
                        fprs = [v["fpr"] for v in sub_eos.values()]
                        c_eo_gap = (max(tprs) - min(tprs)) + (max(fprs) - min(fprs))
                        
                        c_unf = (c_dp_gap + c_eo_gap) / 2
                        
                        # 格式化名称: "race & gender: Male"
                        display_name = f"{key}: {val.lower()}"
                        markdown += f"| {display_name} | {c_dp_gap:.4f} | {c_eo_gap:.4f} | {c_unf:.4f} |\n"

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    print(f"Report successfully generated at: {output_path}")

def main(args):
    # 0. 初始化与数据加载
    print(f">>> [Init] Dataset: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)
    
    pred_df = pd.read_parquet(args.pred_path)
    
    # 当dataset元数据不是csv格式时，你会需要调整这个
    meta_df = pd.read_csv(dataset_config.processed_metadata_path)
    
    full_df = pd.merge(pred_df, meta_df, on="sample_id", how="inner")
    original_count = len(full_df)

    # 1. 过滤无效结果
    valid_df = full_df[full_df['pred_label'] != -1].copy()
    valid_count = len(valid_df)
    unparsable_count = original_count - valid_count
    
    print(f">>> [Clean] Data filtering complete:")
    print(f"    - Original samples: {original_count}")
    print(f"    - Valid samples: {valid_count}")
    print(f"    - Unparsable (Dropped): {unparsable_count} ({unparsable_count/original_count:.2%})")

    # 2. [Process] 动态映射 (基于有效样本)
    eval_df = apply_eval_transforms(valid_df, dataset_config)
    has_prob = 'pred_prob' in eval_df.columns and not eval_df['pred_prob'].isnull().all()
    
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
        
        try:
            # === 核心改动：计算 DP, EO, Unfairness Score ===
            dp_metrics = evaluator.evaluate_demographic_parity(group_keys)
            eo_metrics = evaluator.evaluate_equalized_odds(group_keys)
            unfairness_score = evaluator.calculate_unfairness_score(dp_metrics, eo_metrics)
            
            # 组织 metrics bundle
            metrics_bundle = {
                # [New] 三大核心指标
                "DP_Gap": dp_metrics,
                "EO_Gap": eo_metrics,
                "Unfairness_Score": unfairness_score,
                
                # [Existing] 基础指标 (不改动)
                "accuracy_parity": evaluator.compute_subgroup_performance(group_keys, 'accuracy'),
                "recall_parity": evaluator.compute_subgroup_performance(group_keys, 'recall'), # TPR Parity
                "disparate_impact": evaluator.evaluate_disparate_impact(group_keys)
            }
            
            # [Optional] 概率相关指标
            '''
            if has_prob:
                metrics_bundle["auc_parity"] = evaluator.compute_subgroup_performance(group_keys, 'auc')
                metrics_bundle["calibration_ece"] = evaluator.evaluate_calibration_fairness(group_keys)
            '''
            
            fairness_report[group_name] = metrics_bundle
            
        except KeyError as e:
            continue

    # 6. [Report] 路径处理逻辑 (FIXED)
    if args.output_path:
        # 如果用户指定了路径，直接使用
        report_path = args.output_path
    else:
        # 如果未指定，生成默认路径：data/results/reports/[model]-[dataset]-timestamp_report.json
        # markdown: 
        report_filename = os.path.basename(args.pred_path).replace(".parquet", "_report.json")
        markdown_filename = os.path.basename(args.pred_path).replace(".parquet", "_report.md")
        report_path = os.path.join("data/results/reports", report_filename)
        markdown_report_path = os.path.join("outputs/", markdown_filename)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(report_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    final_output = {
        "meta": {
            "model": args.model_name,
            "dataset": args.dataset,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "original_samples": original_count,
            "valid_samples": valid_count,
            "unparsable_ratio": unparsable_count / original_count,
            "queries": [" & ".join(q) for q in query_combinations]
        },
        "overall_metrics": compute_basic_metrics(eval_df['label'], eval_df['pred_label']),
        "intersectional_analysis": fairness_report
    }
    
    with open(report_path, "w") as f:
        json.dump(final_output, f, indent=4, cls=NpEncoder)
        
    print(f">>> [Report] Successfully saved to: {report_path}")

    # 生成Markdown报告
    generate_markdown_report(report_path, markdown_report_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--model_name", type=str, default="Unknown")
    parser.add_argument("--output_path", type=str, default=None, help="Explicit path to save the report")
    parser.add_argument("--query_attrs", nargs='+', default=None)
    args = parser.parse_args()
    main(args)