import argparse
import pandas as pd
import numpy as np
import json
import os
import time
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super(NpEncoder, self).default(obj)

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    计算多分类指标 (3分类)
    支持 Accuracy, Macro-F1, Macro-Recall
    """
    if len(df) == 0: 
        return {"accuracy": 0.0, "f1_macro": 0.0, "recall_macro": 0.0, "count": 0}
    
    # 过滤无效
    valid = df[df['pred_label'] != -1]
    if len(valid) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0, "recall_macro": 0.0, "count": 0}

    y_true = valid['label']
    y_pred = valid['pred_label']
    
    # 使用 macro average 适合类别不平衡或多分类场景
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "count": len(valid)
    }

def calculate_consistency(base_df: pd.DataFrame, aug_df: pd.DataFrame) -> float:
    """计算一致性"""
    b_valid = base_df[base_df['pred_label'] != -1][['pair_id', 'pred_label']]
    a_valid = aug_df[aug_df['pred_label'] != -1][['pair_id', 'pred_label']]
    
    merged = pd.merge(b_valid, a_valid, on='pair_id', suffixes=('_base', '_aug'))
    if len(merged) == 0: return 0.0
        
    matches = (merged['pred_label_base'] == merged['pred_label_aug']).sum()
    return round(matches / len(merged), 4)

def generate_markdown_report(report_data, output_path):
    model = report_data["meta"]["model"]
    dataset = report_data["meta"]["dataset"]
    
    md = f"# Selection Task Fairness Report\n\n"
    md += f"- **Model**: {model}\n- **Dataset**: {dataset}\n- **Timestamp**: {report_data['meta']['timestamp']}\n\n"
    
    analysis = report_data["analysis"]
    for base_attr, attr_groups in analysis.items():
        md += f"## Base Context: {base_attr.capitalize()}\n\n"
        
        for key, sub_report in attr_groups.items():
            try: add_attr_name = key.split("_perturbed_by_")[-1]
            except: add_attr_name = key
            
            md += f"### Perturbation: {add_attr_name.capitalize()}\n"
            md += "Metrics shown: Accuracy (Overall / Ambig / Disambig) and Macro-F1 (Overall).\n\n"
            
            # 宽表设计：展示 Overall 以及两个子条件的 Acc
            md += "| Context | Overall Acc | Ambig Acc | Disambig Acc | Macro-F1 | Consistency |\n"
            md += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            
            # Helper to safely get nested metric
            def get_m(metrics_dict, condition, metric_name):
                return metrics_dict.get(condition, {}).get(metric_name, "N/A")

            # 1. Base Row
            base_m = sub_report.get("base_metrics", {})
            b_oa = get_m(base_m, "overall", "accuracy")
            b_amb = get_m(base_m, "ambig", "accuracy")
            b_dis = get_m(base_m, "disambig", "accuracy")
            b_f1 = get_m(base_m, "overall", "f1_macro")
            
            md += f"| **Base** | **{b_oa}** | {b_amb} | {b_dis} | {b_f1} | - |\n"
            
            # 2. Augmented Rows
            aug_breakdown = sub_report.get("augmented_breakdown", {})
            for val in sorted(aug_breakdown.keys()):
                m = aug_breakdown[val]
                
                oa = get_m(m, "overall", "accuracy")
                amb = get_m(m, "ambig", "accuracy")
                dis = get_m(m, "disambig", "accuracy")
                f1 = get_m(m, "overall", "f1_macro")
                cons = m.get("consistency", 0.0)
                
                md += f"| + {val} | {oa} | {amb} | {dis} | {f1} | {cons} |\n"
            
            md += "\n> **Gap Analysis**: Difference between Base and Augmented (Positive = Performance Drop)\n"
            md += "---\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

def main(args):
    print(f">>> Loading predictions: {args.pred_path}")
    df = pd.read_parquet(args.pred_path)
    
    report = {
        "meta": {"model": args.model_name, "dataset": args.dataset, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "analysis": {}
    }

    # Helper: 计算三种条件的 metrics
    def calculate_condition_metrics(dataframe):
        return {
            "overall": compute_metrics(dataframe),
            "ambig": compute_metrics(dataframe[dataframe['context_condition'] == 'ambig']),
            "disambig": compute_metrics(dataframe[dataframe['context_condition'] == 'disambig'])
        }

    # 1. Group by Base Attr
    for base_attr, base_group_df in df.groupby("base_attr"):
        attr_report = {}
        
        # 2. Group by Add Attr
        for add_attr, sub_group_df in base_group_df.groupby("add_attr"):
            sub_report = {"base_metrics": {}, "augmented_breakdown": {}, "summary": {}}
            
            # --- Base Data Metrics ---
            base_data = sub_group_df[sub_group_df['pair_type'] == 'base']
            base_metrics_bundle = calculate_condition_metrics(base_data)
            sub_report["base_metrics"] = base_metrics_bundle
            
            # --- Augmented Data Metrics ---
            aug_data_all = sub_group_df[sub_group_df['pair_type'] == 'augmented']
            
            gaps_overall = []
            
            for val, val_data in aug_data_all.groupby("add_attr_value"):
                # 计算该属性值的 Overall/Ambig/Disambig 指标
                aug_metrics_bundle = calculate_condition_metrics(val_data)
                
                # 计算 Gap (仅计算 Overall Acc Gap 用于 Summary，但在 Report 中可以展示更多)
                try:
                    b_acc = base_metrics_bundle["overall"]["accuracy"]
                    a_acc = aug_metrics_bundle["overall"]["accuracy"]
                    acc_gap = round(b_acc - a_acc, 4)
                except:
                    acc_gap = 0.0
                
                # 计算 Consistency
                consistency = calculate_consistency(base_data, val_data)
                
                # 将衍生指标注入到结果中
                aug_metrics_bundle["acc_gap_overall"] = acc_gap
                aug_metrics_bundle["consistency"] = consistency
                
                sub_report["augmented_breakdown"][val] = aug_metrics_bundle
                gaps_overall.append(abs(acc_gap))
            
            sub_report["summary"]["mean_abs_acc_gap"] = round(np.mean(gaps_overall), 4) if gaps_overall else 0.0
            attr_report[f"{base_attr}_perturbed_by_{add_attr}"] = sub_report
            
        report["analysis"][base_attr] = attr_report

    # Save
    os.makedirs("data/results/reports/selection", exist_ok=True)
    out_json_filename = os.path.basename(args.pred_path).replace(".parquet", "_report.json")
    out_json_path = os.path.join("data/results/reports/selection", out_json_filename)

    with open(out_json_path, "w") as f:
        json.dump(report, f, indent=4, cls=NpEncoder)
    
    # 保存 Markdown 报告
    os.makedirs("outputs/selection", exist_ok=True)
    out_md_path = os.path.join("outputs/selection", out_json_filename.replace("_report.json", "_report.md"))
    generate_markdown_report(report, out_md_path)
    
    print(f">>> Evaluation Complete.")
    print(f"    JSON: {out_json_path}")
    print(f"    Markdown: {out_md_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="bbq_selection")
    parser.add_argument("--model_name", type=str, default="Unknown")
    args = parser.parse_args()
    main(args)