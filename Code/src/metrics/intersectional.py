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