# src/data/preprocess_adult.py
import pandas as pd
import numpy as np
import uuid

def process_adult_data(input_path, output_feature_path, output_meta_path):
    # 1. 加载原始逻辑 (复用你提供的清洗代码)
    data = pd.read_csv(input_path, header=0, names=[
        'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ])

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip()

    # 去除缺失值
    data = data.replace('?', np.nan).dropna()

    # **只保留种族为 "Black" 和 "White" 的样本**
    data = data[data['race'].isin(['Black', 'White'])]

    # 将年龄列转换为整数类型
    data['age'] = data['age'].astype(int)

    # 重新编码年龄段
    def encode_age(age):
        if 24 <= age <= 40:
            return '24-40'
        elif 41 <= age <= 60:
            return '41-60'
        elif age >= 61:
            return '61+'
        else:
            return None

    data['age_group'] = data['age'].apply(encode_age)
    data = data.dropna(subset=['age_group'])

    # 选择指定的教育水平
    education_levels = {
        'HS-grad': 'High School',
        'Bachelors': 'Bachelors',
        'Masters': 'Masters'
    }
    data = data[data['education'].isin(education_levels.keys())]
    data['education_level'] = data['education'].map(education_levels)

    # 选择指定的职业
    occupation_mapping = {
        'Adm-clerical': 'Adm-clerical',
        'Exec-managerial': 'Prof-specialty',
        'Prof-specialty': 'Prof-specialty',
        'Sales': 'Sales',
        'Handlers-cleaners': 'Other',
        'Machine-op-inspct': 'Other',
        'Other-service': 'Other',
        'Tech-support': 'Other',
        'Craft-repair': 'Other',
        'Transport-moving': 'Other',
        'Farming-fishing': 'Other',
        'Protective-serv': 'Other',
        'Priv-house-serv': 'Other',
        'Armed-Forces': 'Other'
    }
    data['work_type'] = data['occupation'].map(occupation_mapping)
    data = data.dropna(subset=['work_type'])

    # 过滤工作时长
    data = data[data['hours-per-week'].isin([40, 45, 50])]
    data['work_hours'] = data['hours-per-week'].astype(str)
    
    # -------------------------------------------------
    # 工程化新增：生成唯一 Sample ID
    # -------------------------------------------------
    
    data['sample_id'] = [str(uuid.uuid4()) for _ in range(len(data))]
    
    # 2. 拆分 Feature (模型能看到的) 和 Metadata (评测用的 + 真实标签)
    
    # Feature 只需要包含构建 Prompt 所需的列 + sample_id
    feature_cols = ['sample_id', 'gender', 'race', 'age_group', 'education_level', 'work_hours', 'work_type']
    features = data[feature_cols]
    
    # Metadata 包含敏感属性、真实标签 + sample_id
    # 注意：income 需要转为 0/1
    data['label'] = data['income'].apply(lambda x: 1 if '>50K' in x else 0)
    meta_cols = ['sample_id', 'label', 'gender', 'race', 'age_group'] # 按需保留用于分组分析的列
    metadata = data[meta_cols]
    
    # 3. 保存
    features.to_csv(output_feature_path, index=False)
    metadata.to_csv(output_meta_path, index=False)
    print(f"Processed data saved to {output_feature_path} and {output_meta_path}")

# 注意：这个函数应该被一个独立的脚本调用，比如 scripts/prepare_data.py