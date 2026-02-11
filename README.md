<<<<<<< HEAD
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
=======
# Intersectional\_Fairness\_alpha

**A Modular Framework for Evaluating Large Language Model Intersectional Fairness**

本项目旨在建立一个用于评估大语言模型（LLMs）在**交叉公平性（Intersectional Fairness）上的调研与评测框架。不同于单一维度的公平性评估，本框架专注于捕捉多重属性（如“黑人+女性+高龄”）组合下的涌现性偏见（Emergent Bias）**。

设计理念参考了 **OpenCompass** 和 **CEB** 的架构，强调数据流解耦、配置化管理以及学术级指标的全面性。

---

## 核心特性 (Key Features)

* **解耦架构 (Decoupled Pipeline)**: 将推理（Generation）与评估（Evaluation）完全分离。推理阶段只接触脱敏特征，评估阶段结合元数据（Metadata）进行分析，确保测试的纯粹性与隐私安全。
* **动态交叉查询 (Dynamic Intersectional Query)**: 支持通过 Query 列表（如 `['race', 'gender']`）动态生成交叉子群，无需硬编码分组逻辑。
* **多维评估指标 (Comprehensive Metrics)**: 不仅包含传统的 Demographic Parity，还集成了 Differential Fairness ($\epsilon$-fairness)、Calibration Error (ECE)、Disparate Impact 等高级指标。
* **配置驱动 (Config-Driven)**: 支持通过配置文件快速接入新数据集（Adult, COMPAS, German等），自动处理字段映射与 Prompt 构建。
* **标准化中间件**: 采用 `Parquet` 格式存储带有置信度（Probability）的中间推理结果，支持断点续传与高效读取。

---

## 仓库结构 (Directory Structure)

```text
code/
├── configs/
│   ├── __init__.py             # 动态配置加载器
│   ├── adult.py                # Adult 数据集配置 (字段映射, Prompt模板)
│   └── [dataset].py            # 扩展其他数据集
├── data/
│   ├── raw/                    # 原始 CSV/JSON 数据
│   ├── processed/              # 预处理后的数据 (清洗完毕)
│   │   ├── [dataset]_features.csv   # 模型输入: sample_id + 文本特征
│   │   └── [dataset]_metadata.csv   # 评测真值: sample_id + 敏感属性 + GT
│   └── results/
│       ├── predictions/        # 中间结果 (Parquet): pred_label, pred_prob
│       └── reports/            # 最终报告 (JSON): 包含所有交叉公平性指标
├── src/
│   ├── data/
│   │   ├── preprocess_adult.py # 数据清洗逻辑
│   ├── models/
│   │   ├── huggingface.py      # HF 模型封装 (支持 Logits 提取)
│   ├── metrics/
│   │   ├── functional.py       # 基础指标 (Acc, F1, Recall)
│   │   └── intersectional.py   # 核心: 交叉分组逻辑与高级公平性计算
├── run_gen.py                  # 阶段 I: 推理 (LLM -> Parquet)
├── run_eval.py                 # 阶段 II: 评估 (Parquet + Metadata -> Report)
├── run.sh                      # 一键运行脚本
└── prepare_data.py             # 数据准备入口
```

---

## 评估方法论 (Methodology)

### 1\. 分类任务评估流程

我们采用 **“生成-查询-度量”** 的三步走策略：

1. **特征提取与 Prompt 构建**:
   假定数据集 $A$ 中有样本集合 $X$。对于每一个 $x_i$，我们提取其非敏感属性 $a_j$ 构建 Prompt，令模型完成二分类预测（如：“预测年收入是否大于50k”）。

   * *Input*: `{age_group: '24-40', education: 'Bachelors', ...}`
   * *Output*: `y_pred` (Label) & `y_prob` (Confidence)
2. **中间结果存储**:
   系统将生成结果与原始属性解耦存储。评估时，通过 `sample_id` 将预测结果与 `Metadata` 表中的敏感属性（如种族、性别）对齐。

   > **数据对齐表项格式**:
   > `Item: {sample_id, a_1:val_1, ..., a_n:val_n, y_pred, y_real, y_prob}`
   >
3. **交叉查询 (Intersectional Query)**:
   评估器接受一个敏感属性集合 $S = \{a_{j1}, a_{j2}, ..., a_{jn}\}$ 作为 Query。

   * *Query*: `['race', 'gender']`
   * *Action*: 系统自动生成所有可能的组合（如 `Black-Female`, `White-Male`...），并计算子群间的差异。

### 2\. 属性名对齐 (Attribute Alignment)

为了最小化跨数据集评测的迁移成本，我们将不同数据集的字段映射到以下标准集合：

* `gender`: 性别 (Male, Female, ...)
* `age_group`: 年龄分桶 (24-40, 40-60, 60+)
* `race`: 种族
* `edu_level`: 教育程度
* `work_class`: 工作性质

---

## 支持指标 (Supported Metrics)

本框架在 `src/metrics/intersectional.py` 中实现了以下指标体系：

| 类别                  | 指标名称                                       | 描述                                      | 适用场景                           |
| :-------------------- | :--------------------------------------------- | :---------------------------------------- | :--------------------------------- |
| **基础性能**    | Accuracy / F1                                  | 总体及各子群的预测准确性                  | 通用                               |
| **基础公平性**  | **Demographic Parity**                   | 正例预测率 (PPR) 的差异                   | 关注结果均等                       |
|                       | **Equalized Odds**                       | TPR (Recall) 和 FPR 的差异                | 关注错误率均等                     |
| **交叉公平性**  | **Differential Fairness ($\epsilon$)** | 任意两子群间预测概率比的对数最大值        | 严格的交叉偏见检测                 |
|                       | **Disparate Impact**                     | Min(PPR) / Max(PPR)                       | 法律合规性 (80% Rule)              |
|                       | **Worst-case Accuracy**                  | 表现最差子群的准确率                      | 关注短板效应                       |
| **高级/置信度** | **Group ECE**                            | 群体校准误差 (Expected Calibration Error) | 衡量模型是否对某些群体“盲目自信” |
|                       | **Treatment Equality**                   | FN/FP 比率的差异                          | 衡量模型在不同群体上的错误倾向     |

---

## 快速开始 (Quick Start)

### 1\. 环境准备

```bash
pip install -r requirements.txt
# 主要依赖: torch, transformers, pandas, pyarrow, scikit-learn
```

### 2\. 数据准备

将原始数据（如 `adult.csv`）放置在 `data/raw/` 目录下，运行清洗脚本：

```bash
python prepare_data.py
```

此步骤会生成 `data/processed/` 下的 feature 和 metadata 文件。

### 3\. 一键运行 (Pipeline)

参考 `Scripts/gen_adult_dataset.sh` 中的写法，直接编辑 `run.sh` 设置模型路径和数据集名称，然后运行：

```bash
bash run.sh
```

**或者分步运行：**

* **Step 1: 模型推理**

  ```bash
  python run_gen.py \
      --dataset adult \
      --model_path "Qwen/Qwen2.5-1.5B-Instruct" \
      --batch_size 8
  ```

  *输出*: `data/results/predictions/Qwen2.5-1.5B-adult-[timestamp].parquet`
* **Step 2: 公平性评估**

  ```bash
  python run_eval.py \
      --pred_path "data/results/predictions/..." \
      --dataset adult
  ```

  *输出*: `data/results/reports/..._report.json`

---

## 配置指南 (Configuration)

如需添加新数据集（例如 COMPAS），请在 `code/configs/` 下新建 `compass.py`：

```python
from dataclasses import dataclass, field

@dataclass
class CompassConfig:
    dataset_name: str = "compass"
    # 定义 CSV 列名到标准属性名的映射
    column_mapping: Dict[str, str] = field(default_factory=lambda: {
        "gender": "sex",
        "race": "race_code",
        # ...
    })
    # 定义 Prompt 模板
    prompt_template: str = "..."
    # 定义用于交叉评估的敏感属性列表
    sensitive_attributes: List[str] = field(default_factory=lambda: ["race", "gender"])

dataset_config = CompassConfig()
```

然后在脚本中通过 `--dataset compass` 调用即可。

---
>>>>>>> ac22061 (sync latest local version)
