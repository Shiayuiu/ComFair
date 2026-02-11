## src/data/preprocess_adult.py
import pandas as pd
import numpy as np
import uuid

def process_adult_data(input_path, output_feature_path, output_meta_path, sample_size=None):
    # 1. 读取数据
    cols = [
        'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    data = pd.read_csv(input_path, header=0, names=cols)

    # 基础清洗
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip()
    data = data.replace('?', np.nan).dropna()

    # 过滤种族 (保持你的实验设置)
    data = data[data['race'].isin(['Black', 'White'])]

    # sampling
    if sample_size:
        data = data.sample(n=sample_size, random_state=42)
    
    # 生成唯一 ID
    data['sample_id'] = [str(uuid.uuid4()) for _ in range(len(data))]
    
    # 2. 准备 Feature 表 (全量原始特征)
    # 排除掉作为 label 的 income，其余全是特征
    feature_cols = ['sample_id'] + [c for c in cols if c != 'income' and c != 'fnlwgt']
    features = data[feature_cols]
    
    # 3. 准备 Metadata 表 (包含用于评估的原始列 + Label)
    data['label'] = data['income'].apply(lambda x: 1 if '>50K' in x else 0)
    meta_cols = ['sample_id', 'label'] + [c for c in cols if c != 'income']
    metadata = data[meta_cols]
    
    # 保存结果
    features.to_csv(output_feature_path, index=False)
    metadata.to_csv(output_meta_path, index=False)
    print(f"Preprocess Done: Features saved to {output_feature_path}")