import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import entropy, chi2_contingency
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings

class IntersectionalEvaluator:
    """
    交叉公平性评估器 (Refactored)
    结合了工程化分片逻辑与丰富的公平性指标体系。
    """
    def __init__(self, df: pd.DataFrame, label_col: str = "label", pred_col: str = "pred_label", prob_col: str = None):
        """
        Args:
            df: 包含预测结果和元数据的大宽表
            label_col: 真实标签列名
            pred_col: 预测标签列名
            prob_col: 预测概率列名 (用于AUC计算，可选)
        """
        self.df = df.copy()
        self.label_col = label_col
        self.pred_col = pred_col
        self.prob_col = prob_col
        
        # 基础验证
        if self.label_col not in self.df.columns or self.pred_col not in self.df.columns:
            raise ValueError(f"Missing columns. Required: {self.label_col}, {self.pred_col}")

    # =========================================================================
    # Core Logic: Subgroup Extraction (保留架构)
    # =========================================================================
    def _get_subgroup_data(self, group_keys: List[str], min_samples: int = 30) -> List[Dict]:
        """
        核心分片函数：动态根据输入的属性列表生成子群
        """
        # 填充空值以防丢失数据
        safe_keys = [self.df[k].fillna("Unknown") for k in group_keys]
        
        # 分组
        try:
            grouped = self.df.groupby(safe_keys)
        except KeyError as e:
            raise KeyError(f"Grouping keys not found in DataFrame: {e}")

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
            
        if len(valid_subgroups) < 2:
            warnings.warn(f"Valid subgroups < 2 for keys {group_keys}. Fairness metrics may be meaningless.")
            
        return valid_subgroups

    # =========================================================================
    # Metric Implementations (对接你的指标需求)
    # =========================================================================

    # --- Interface: Basic Stats (Accuracy, Precision, Recall, F1, AUC) ---
    def compute_subgroup_performance(self, group_keys: List[str], metric: str = 'accuracy') -> Dict:
        """
        通用性能计算入口，支持: accuracy, precision, recall, f1, auc
        对应你原本的: calculate_group_accuracy, calculate_group_recall 等
        """
        subgroups = self._get_subgroup_data(group_keys)
        details = {}
        scores = []

        for g in subgroups:
            val = 0.0
            if metric == 'accuracy':
                val = accuracy_score(g['y_true'], g['y_pred'])
            elif metric == 'precision':
                val = precision_score(g['y_true'], g['y_pred'], zero_division=0)
            elif metric == 'recall':
                val = recall_score(g['y_true'], g['y_pred'], zero_division=0)
            elif metric == 'f1':
                val = f1_score(g['y_true'], g['y_pred'], zero_division=0)
            elif metric == 'auc':
                # AUC 特殊处理：如果只有一个类别，无法计算
                if len(np.unique(g['y_true'])) < 2 or g['y_prob'] is None:
                    val = 0.5 # Default fallback
                else:
                    try:
                        val = roc_auc_score(g['y_true'], g['y_prob'])
                    except:
                        val = 0.5
            
            details[g['group_id']] = round(float(val), 4)
            scores.append(val)

        # 整合多种差异度量
        return self._summarize_comprehensive(metric, scores, details)

    # --- Interface: Core Fairness Metrics ---
    
    def evaluate_equal_opportunity(self, group_keys: List[str]) -> Dict:
        """
        对应: equal_opportunity_difference
        逻辑: 比较各组 True Positive Rate (Recall) 的差异
        """
        return self.compute_subgroup_performance(group_keys, metric='recall')

    def evaluate_predictive_parity(self, group_keys: List[str]) -> Dict:
        """
        对应: predictive_parity_difference
        逻辑: 比较各组 Precision 的差异
        """
        return self.compute_subgroup_performance(group_keys, metric='precision')

    def evaluate_worst_case(self, group_keys: List[str], metric: str = 'accuracy') -> Dict:
        """
        对应: worst_subgroup_accuracy
        """
        stats = self.compute_subgroup_performance(group_keys, metric)
        return {
            "metric": metric,
            "worst_group": min(stats['subgroup_details'], key=stats['subgroup_details'].get),
            "worst_score": stats['min_score']
        }

    # --- Interface: Inequality Metrics (Gini, Entropy) ---
    # 这是你代码中比较 Fancy 的部分，我实现了具体逻辑

    def _gini_coefficient(self, values: List[float]) -> float:
        """辅助函数：计算基尼系数"""
        if not values: return 0.0
        # Gini = sum(|xi - xj|) / (2 * n^2 * mean)
        # 这里我们计算指标分布的不平等度（指标越接近代表越公平，Gini越小）
        sorted_vals = sorted(values)
        n = len(values)
        accumulated = np.cumsum(sorted_vals)
        if accumulated[-1] == 0: return 0.0
        return (n + 1 - 2 * np.sum(accumulated) / accumulated[-1]) / n

    def evaluate_distributional_fairness(self, group_keys: List[str], metric: str = 'accuracy') -> Dict:
        """
        对应: intersectional_gini_coefficient, entropy_fairness_index, theil_index
        计算某个性能指标在不同群体间分布的不平等程度
        """
        subgroups = self._get_subgroup_data(group_keys)
        
        # 计算每个组的性能指标
        scores = []
        for g in subgroups:
            if metric == 'accuracy':
                val = accuracy_score(g['y_true'], g['y_pred'])
            else:
                val = f1_score(g['y_true'], g['y_pred'], zero_division=0) # Default to F1
            scores.append(val)
        
        if not scores:
            return {}

        # 1. Gini
        gini = self._gini_coefficient(scores)
        
        # 2. Entropy (这里计算分布的熵，值越大说明分布越均匀/公平)
        # Normalize scores to probability distribution for entropy calc
        score_sum = sum(scores)
        if score_sum > 0:
            prob_dist = [s/score_sum for s in scores]
            ent_val = entropy(prob_dist)
        else:
            ent_val = 0.0

        return {
            "target_metric": metric,
            "gini_coefficient": round(gini, 4), # 越低越好 (0=完全平等)
            "entropy_index": round(ent_val, 4), # 越高越好
            "mean_performance": round(float(np.mean(scores)), 4)
        }

    # --- Interface: Statistical Significance (Optional) ---
    
    def evaluate_statistical_significance(self, group_keys: List[str]) -> Dict:
        """
        对应: statistical_significance_test
        这里实现一个简单的卡方独立性检验 (Chi-Square Test of Independence)
        检验 'Group Membership' 和 'Error Rate' 是否独立
        """
        subgroups = self._get_subgroup_data(group_keys)
        
        # 构建列联表 (Contingency Table)
        # Rows: Groups, Cols: [Correct, Incorrect]
        observed = []
        group_names = []
        
        for g in subgroups:
            correct = np.sum(g['y_true'] == g['y_pred'])
            incorrect = len(g['y_true']) - correct
            observed.append([correct, incorrect])
            group_names.append(g['group_id'])
            
        if len(observed) < 2:
            return {"p_value": None, "significance": "Not enough groups"}

        try:
            chi2, p, dof, expected = chi2_contingency(observed)
            is_significant = p < 0.05
        except:
            return {"error": "Chi-square calculation failed"}

        return {
            "test_type": "Chi-Square Test of Independence",
            "p_value": float(p), # 如果 p < 0.05，说明不同组的错误率分布显著不同（即存在不公平）
            "is_significant_disparity": bool(is_significant),
            "dof": dof
        }

    # =========================================================================
    # Helpers & Summarizers
    # =========================================================================

    def _summarize_comprehensive(self, metric_name: str, scores: List[float], details: Dict) -> Dict:
        """
        统一的输出格式化，涵盖了你要求的 Disparity Ratio, Gap, etc.
        """
        if not scores:
            return {"error": "No valid groups"}
            
        _min = min(scores)
        _max = max(scores)
        _mean = np.mean(scores)
        
        return {
            "metric": metric_name,
            "subgroup_details": details,
            # 基础统计
            "min_score": round(_min, 4),
            "max_score": round(_max, 4),
            "mean_score": round(_mean, 4),
            "std_dev": round(float(np.std(scores)), 4),
            
            # 对应: accuracy_disparity_ratio (Min/Max)
            # 越接近 1 越好
            "disparity_ratio": round(_min / _max, 4) if _max > 0 else 0.0,
            
            # 对应: intersectional_mean_gap (Max - Min)
            # 越接近 0 越好
            "max_gap": round(_max - _min, 4),
        }

    # =========================================================================
    # Unimplemented / Redundant Metrics (Marked as requested)
    # =========================================================================
    
    def dynamic_range(self, groups: Dict):
        """
        [User Request]: dynamic_range
        [Status]: Pass / Redundant.
        Reason: Included in 'max_gap' (Max - Min) in summary.
        """
        pass 

    def z_score_difference(self, groups: Dict):
        """
        [User Request]: z_score_difference
        [Status]: Not Implemented.
        Reason: Requires robust population mean/std estimation which is unstable for small subgroups.
        """
        pass

    def confidence_intervals(self, groups: Dict):
        """
        [User Request]: confidence_intervals
        [Status]: Future Work.
        Reason: Requires bootstrapping, computationally expensive for real-time eval.
        """
        pass