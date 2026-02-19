<<<<<<< HEAD
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

class IntersectionalEvaluator:
    def __init__(self, data: pd.DataFrame, sensitive_attributes: list, min_group_size=30):
        """
        - data : DataFrame
        - sensitive_attributes : list of attrs
        - min_group_size : valid group size
        """
        self.data = data.copy()
        self.sensitive_attributes = sensitive_attributes
        self.min_group_size = min_group_size

        required_cols = ['sample_id', 'y_true', 'y_pred'] + sensitive_attributes
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f'Missing cols in data: {missing}.')
        
    def evaluate(self, attribute_combination, metrics):
        """
        1. gen X-group
        2. use filter
        3. compute
        """

        self.data['intersectional_group'] = self.data[attribute_combination].astype(str).agg('_'.join, axis=1)

        group_counts = self.data['intersectional_group'].value_counts()
        valid_groups = group_counts[group_counts >= self.min_group_size].index.tolist()
        invalid_groups = group_counts[group_counts < self.min_group_size].index.tolist()

        if len(valid_groups) < 2:
            raise ValueError(
                    f'有效交集组数量不足, 当前数量:{len(valid_groups)}, 组名: {valid_groups}'
                )
        
        results = {
            'all_groups': self.data['intersectional_group'].unique().tolist(),
            'valid_groups': valid_groups,
            'invalid_groups': invalid_groups 
        }

        # 这里可以改为计算多个指标然后复合评分的逻辑
        for metric in metrics:
            if metric == 'accuracy':
                results[metric] = self._compute_accuracy(valid_groups)
            elif metric == 'equalized_odds':
                results[metric] = self._compute_equalized_odds(valid_groups)
            elif metric == 'auc':
                results[metric] = self._compute_auc(valid_groups)
            # 可补充其余指标


        return results
    
    def _compute_accuracy(self, valid_groups):
        group_metrics = {}
        for group in valid_groups:
            subset = self.data[self.data['intersectional_group'] == group]
            acc = (subset['y_true'] == subset['y_pred']).mean()
            group_metrics[group] = float(acc) # JSON序列化保证

        values = list(group_metrics.values())
        return {
            "per_group": group_metrics,
            "fairness_gap": max(values) - min(values),
            "std_dev": float(np.std(values))
        }
    
    def _compute_equalized_odds(self, valid_groups):
        """计算TPR/FPR组间差异"""
        tpr_fpr = {}
        for group in valid_groups:
            subset = self.data[self.data["intersectional_group"] == group]
            tn, fp, fn, tp = confusion_matrix(
                subset["y_true"], subset["y_pred"]
            ).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr_fpr[group] = {"TPR": float(tpr), "FPR": float(fpr)}
        
        tpr_values = [v["TPR"] for v in tpr_fpr.values()]
        fpr_values = [v["FPR"] for v in tpr_fpr.values()]
        return {
            "per_group": tpr_fpr,
            "tpr_gap": max(tpr_values) - min(tpr_values),
            "fpr_gap": max(fpr_values) - min(fpr_values)
        }
    
    def _compute_auc(self, valid_groups):
        return
=======
# src/metrics/intersectional.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import entropy
from typing import Dict, List, Any, Optional
import warnings

class IntersectionalEvaluator:
    """
    交叉公平性评估器
    负责根据给定的 Group Keys 对 DataFrame 进行切片，并计算各项指标。
    """
    def __init__(self, df: pd.DataFrame, label_col: str = "label", pred_col: str = "pred_label", prob_col: str = None):
        self.df = df.copy()
        self.label_col = label_col
        self.pred_col = pred_col
        self.prob_col = prob_col
        
        # 检查
        if self.label_col not in self.df.columns or self.pred_col not in self.df.columns:
            raise ValueError(f"Missing columns. Required: {self.label_col}, {self.pred_col}")

    def _get_subgroup_data(self, group_keys: List[str], min_samples: int = 15) -> List[Dict]:
        """
        核心切片逻辑：根据传入的 keys (例如 ['race', 'age_mapped']) 进行分组
        """
        # 填充空值
        safe_keys = [self.df[k].fillna("Unknown") for k in group_keys]
        
        try:
            grouped = self.df.groupby(safe_keys)
        except KeyError as e:
            raise KeyError(f"Grouping keys not found in DataFrame: {e}. Did you apply transforms?")

        valid_subgroups = []
        for name, group in grouped:
            if len(group) < min_samples:
                continue
            
            # 格式化组名
            group_name = name if isinstance(name, tuple) else (name,)
            group_id = " & ".join([str(x) for x in group_name])
            
            valid_subgroups.append({
                "group_id": group_id,
                "y_true": group[self.label_col].values,
                "y_pred": group[self.pred_col].values,
                "y_prob": group[self.prob_col].values if self.prob_col and self.prob_col in group else None,
                "size": len(group)
            })
            
        return valid_subgroups

    # --- 基础性能计算 ---
    def compute_subgroup_performance(self, group_keys: List[str], metric: str = 'accuracy') -> Dict:
        subgroups = self._get_subgroup_data(group_keys)
        details = {}
        scores = []

        for g in subgroups:
            val = 0.0
            if metric == 'accuracy':
                val = accuracy_score(g['y_true'], g['y_pred'])
            elif metric == 'recall': # TPR
                val = recall_score(g['y_true'], g['y_pred'], zero_division=0)
            elif metric == 'precision':
                val = precision_score(g['y_true'], g['y_pred'], zero_division=0)
            elif metric == 'auc':
                # AUC 特殊处理：如果只有一个类别，无法计算
                if len(np.unique(g['y_true'])) < 2 or g['y_prob'] is None:
                    val = 0.5 
                else:
                    try:
                        val = roc_auc_score(g['y_true'], g['y_prob'])
                    except:
                        val = 0.5
            
            details[g['group_id']] = round(float(val), 4)
            scores.append(val)

        return self._summarize_stats(metric, scores, details)

    # --- 高级公平性指标 ---

    def evaluate_disparate_impact(self, group_keys: List[str]) -> Dict:
        """Min(PPR) / Max(PPR)"""
        subgroups = self._get_subgroup_data(group_keys)
        ppr_values = []
        details = {}
        for g in subgroups:
            ppr = np.mean(g['y_pred'])
            ppr_values.append(ppr)
            details[g['group_id']] = round(ppr, 4)
            
        if not ppr_values or max(ppr_values) == 0:
            return {"error": "All PPRs are zero"}
            
        return {
            "metric": "Disparate Impact",
            "score": round(min(ppr_values) / max(ppr_values), 4),
            "subgroup_details": details
        }
    
    def evaluate_demographic_parity(self, group_keys: List[str]) -> Dict:
        """
        [DP Gap] 计算 Demographic Parity Gap
        Formula: Max(PPR) - Min(PPR)
        """
        subgroups = self._get_subgroup_data(group_keys)
        ppr_values = []
        details = {}
        
        for g in subgroups:
            ppr = np.mean(g['y_pred'])
            ppr_values.append(ppr)
            details[g['group_id']] = round(ppr, 4)
            
        if not ppr_values: return {"error": "No groups"}
        
        gap = max(ppr_values) - min(ppr_values)
        
        return {
            "metric": "Demographic Parity Gap (DP)",
            "value": round(gap, 4), # 这就是 DP 值
            "max_ppr": round(max(ppr_values), 4),
            "min_ppr": round(min(ppr_values), 4),
            "details": details
        }

    def evaluate_equalized_odds(self, group_keys: List[str]) -> Dict:
        """
        [EO Gap] 计算 Equalized Odds Gap
        Formula: (Max_TPR - Min_TPR) + (Max_FPR - Min_FPR)
        """
        subgroups = self._get_subgroup_data(group_keys)
        tpr_values = []
        fpr_values = []
        details = {}
        
        for g in subgroups:
            tn, fp, fn, tp = confusion_matrix(g['y_true'], g['y_pred'], labels=[0, 1]).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
            details[g['group_id']] = {"tpr": round(tpr, 4), "fpr": round(fpr, 4)}
            
        if not tpr_values: return {"error": "No groups"}
        
        tpr_gap = max(tpr_values) - min(tpr_values)
        fpr_gap = max(fpr_values) - min(fpr_values)
        eo_gap = tpr_gap + fpr_gap
        
        return {
            "metric": "Equalized Odds Gap (EO)",
            "value": round(eo_gap, 4), # 这就是 EO 值
            "tpr_gap": round(tpr_gap, 4),
            "fpr_gap": round(fpr_gap, 4),
            "details": details
        }

    def calculate_unfairness_score(self, dp_result: Dict, eo_result: Dict) -> float:
        """
        [Composite Score] 计算综合不公平得分
        Formula: (DP + EO) / 2
        """
        try:
            dp = dp_result.get("value", 0.0)
            eo = eo_result.get("value", 0.0)
            # 简单平均，也可以根据需求加权
            score = (dp + eo) / 2
            return round(score, 4)
        except:
            return -1.0

    def evaluate_calibration_fairness(self, group_keys: List[str], n_bins: int = 10) -> Dict:
        """Group ECE"""
        subgroups = self._get_subgroup_data(group_keys)
        ece_scores = {}
        for g in subgroups:
            if g['y_prob'] is None: return {"error": "Missing probabilities"}
            
            prob_true, prob_pred = g['y_true'], g['y_prob']
            ece = 0.0
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            
            for i in range(n_bins):
                mask = (prob_pred >= bin_boundaries[i]) & (prob_pred < bin_boundaries[i+1])
                bin_count = np.sum(mask)
                if bin_count > 0:
                    acc = np.mean(prob_true[mask] == (prob_pred[mask] > 0.5))
                    conf = np.mean(prob_pred[mask])
                    ece += (bin_count / len(prob_true)) * np.abs(acc - conf)
            ece_scores[g['group_id']] = round(ece, 4)
            
        values = list(ece_scores.values())
        return self._summarize_stats("Group ECE", values, ece_scores)

    def _summarize_stats(self, metric_name, scores, details):
        if not scores: return {"error": "No valid groups"}
        return {
            "metric": metric_name,
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "mean": round(np.mean(scores), 4),
            "disparity_gap": round(max(scores) - min(scores), 4), # Max - Min
            "subgroup_details": details
        }
>>>>>>> ac22061 (sync latest local version)
