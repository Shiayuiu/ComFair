<<<<<<< HEAD
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
=======
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
        report_path = os.path.join("data/results/reports/classification", report_filename)
        markdown_report_path = os.path.join("outputs/classification/", markdown_filename)
    
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
>>>>>>> ac22061 (sync latest local version)
