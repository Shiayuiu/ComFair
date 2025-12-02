import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix

# 检查
class ClassificationEvaluator:
    def __init__(self, data: pd.DataFrame, sensitive_attributes: list, min_group_size=30):

        self.data = data.copy()
        
        required_cols = ['sample_id', 'y_true', 'y_pred'] + sensitive_attributes
        
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f'Missing cols in data: {missing}.')