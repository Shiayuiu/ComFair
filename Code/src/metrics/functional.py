# src/metrics/functional.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def compute_basic_metrics(y_true, y_pred):
    """计算一组基础指标"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0), # 即 TPR
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

def compute_positive_rate(y_pred):
    """计算正例预测率 (PPR) - 用于 Demographic Parity"""
    if len(y_pred) == 0: return 0.0
    return np.mean(y_pred)

def compute_tpr_fpr(y_true, y_pred):
    """计算 TPR 和 FPR - 用于 Equalized Odds"""
    # tn, fp, fn, tp
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        # 处理只有一种 label 的极端情况
        return {"tpr": 0.0, "fpr": 0.0}
        
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {"tpr": tpr, "fpr": fpr}