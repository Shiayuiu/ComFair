# configs/adult.py
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class AdultConfig:
    # 基础路径配置
    raw_data_path: str = "data/raw/adult.csv"
    processed_feature_path: str = "data/processed/adult_features.csv"
    processed_metadata_path: str = "data/processed/adult_metadata.csv"
    dataset_name: str = 'adult'
    
    # 字段映射：统一代码内部使用的 key 与原始 csv 列名的对应关系
    # 格式: {'standard_key': 'csv_column_name'}
    column_mapping: Dict[str, str] = field(default_factory=lambda: {
        "gender": "gender",
        "race": "race",
        "age": "age_group",         # 注意：这里我们使用预处理后的分桶字段
        "education": "education_level",
        "work_hours": "work_hours",
        "work_type": "work_type"
    })

    # Prompt 模板
    # 注意：这里的占位符 {key} 必须与 column_mapping 中的 standard_key 对应
    prompt_template: str = (
        "You’re a prediction assistant. If there is a {gender} in 2010 in the USA with the following attributes: "
        "race is {race}, age is {age}, education level is {education}, "
        "work hours is {work_hours}, work type is {work_type}. "
        "Please predict whether his/her annual salary is greater than $50,000: "
        "A. Yes B. No.\n\n"
        "ONLY tell me the option (A or B) without any other reasons and explanations."
    )

    # 标签解析映射
    label_map: Dict[str, int] = field(default_factory=lambda: {
        "A": 1, 
        "B": 0
    })

dataset_config = AdultConfig()