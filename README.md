# Intersectional_Fairness_alpha

本代码仓库用于存储以Intersectional_Fairness为主题的调研结果，code和工作流

1\. 调研

关于模型文本生成的公平性指标，关于图像生成的公平性指标；

对于常见任务（生成任务、分类任务等）的评测标准；

如何建立一个区分度的Intersectional-fairness的指标

2\.code demo

代码方案：采用CEB、Opencompass的repo作为参考。

Opencompass的框架：数据集、模型（开源、闭源）接口；对于生成类任务，分别保留infer、eval阶段的数据（将infer阶段的数据保留，可以通过自行接入的evaluator得到评测结果；对于默认的结果，生成指标的表格。

目前已经实现的指标在src/metrics/目录下的Intersectional.py文件中,对于需要query查询的指标目前正在本地测试pipeline。

仓库结构；
```
code/
├── configs/
│   └── [dataset].py             # 任务配置 + 敏感属性定义
├── data/
│   ├── raw/                     # 原始数据
│   ├── processed/               # 预处理数据
│   │   ├── [dataset]_features.csv    # 任务特征 + sample_id
│   │   └── [dataset]_metadata.csv    # 敏感属性元数据 + sample_id
│   └── results/
│       ├── predictions/         # 目录，按模型-数据集-时间戳组织
│       │   └── [model]-[dataset]-[timestamp].parquet
│       └── reports/             # 目录，对应预测结果
│           └── [model]-[dataset]-[timestamp].json
├── src/
│   ├── data/
│   │   ├── loader.py           # 安全数据加载
│   │   └── preprocess.py       # 属性隔离预处理
│   ├── models/
│   │   └── __init__.py         # 模型接口规范
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── classification.py   # 基础指标
│   │   ├── intersectional.py   # 交叉公平性扩展
│   │   └── group_validation.py # 群体有效性检查
│   └── utils/
│       ├── validation.py       # 数据验证
│       └── reporting.py        # 报告生成
├── run_gen.py                  # 生成预测 A
├── run_eval.py                 # 交叉公平性评估 B 
└── run_full.py                 # 全流程评估 A+B
```
