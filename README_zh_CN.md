# README for Beyond Single-Attribute Auditing: A Configurable Multi-Attribute Benchmark for Fairness in LLMs

# ComFair Library

本仓库提供了 `ComFair` ，一个可配置的用于评估大型语言模型在不同任务上的交叉公平性（intersectional fairness）的Benchmark. 下面说明代码组织、环境依赖、运行方法、数据集结构与常见目录说明，帮助您快速实现评测。

## **主要内容与核心特性**

* **解耦架构 (Decoupled Pipeline)**: 将推理（Generation）与评估（Evaluation）完全分离。
* **动态交叉查询 (Dynamic Intersectional Query)**: 支持通过 Query 列表（如 `['race', 'gender']`）动态生成交叉子群体进行评估，无需硬编码分组逻辑。
* **多维评估指标 (Comprehensive Metrics)**: 通过基础指标和交叉公平性指标，ComFair能够给出一个模型的多维评分。
* **配置驱动 (Config-Driven)**: 支持通过用户自行导入配置文件快速接入新数据集，实现自由的评测扩展。

## **快速开始**

### 环境配置

1. 克隆仓库并进入目录：

```bash
git clone https://github.com/Shiayuiu/Intersectional_Fairness_alpha ComFair
cd ComFair
```

2. 使用 conda 创建虚拟环境：

```bash
conda create -n ComFair python=3.10 -y
conda activate ComFair
pip install -r requirements.txt
```

如果使用其它环境管理工具（如 `venv`、`pipenv`、`poetry`），请按相应方式安装 `requirements.txt` 中列出的依赖。

注意：如果您仅需要进行API类模型的评测而不需要在本地部署模型，请忽略 `requirements.txt`中关于CUDA的部分

### 数据准备

我们的数据已经在Huggingface上开源，您可以通过如下的链接访问我们的数据，然后将其下载并存放到本仓库的 `Code/data/` 文件目录下，以使得我们的评估器能够正确读取文件。

[ComFair: A Configurable Multi-Attribute Benchmark](https://huggingface.co/datasets/benchmark-lab/Intersectional_Fairness_alpha_v2/tree/main "ComFair: Configurable Multi-Attribute Benchmark ")

注意：数据路径被我们硬编码在了 `Code/configs/` 文件夹的对应路径中，如果你需要自行指定数据的存放位置，请修改config文件中的 `data_dir` 参数

您需要在 `Code/data/` 目录下新建 `processed/` 目录以保证config文件的一致性，在这一步之后，您的 `data/` 目录应当具有如下的形式：

```
data/
└── processed/              # 处理后的数据, 即我们提供的数据
    ├── recognition/
    ├── ...
    └── selection/
```

### **运行方法**

我们采用Generation和Evaluation两个步骤来评测您的模型，所有任务对应的python文件位于  `Code/` 目录下，形如 `run_gen_[task_name].py` 和 `run_eval_[task_name].py` . 您可以直接使用命令行方式评测，但我们更推荐您使用脚本评测，以便于指定需要的参数。为了方便您理解脚本的结构，我们在 `Code/scripts/` 目录下保留了我们在评测模型时所用的脚本，并提供了两个示例脚本和相关说明，供您参考。

**Step1: Generation**

在 `Code/scripts/` 目录下有我们提供的示例，在这个脚本中，我们指定了：

* 数据集
* 评估维度，即我们针对多少个交叉属性
* 基准属性，即我们考察哪个属性所受的交叉公平影响，用于显式匹配数据集文件
* 模型类型：API模型
* 模型名称以及模型的标识（如 `deepseek-chat` ）
* Base_url以及API_key
* 一些其余参数

运行如下的命令，模型会自行读取对应数据集中的config文件并运行生成，将中间结果存放在 `Code/data/results/predicitons/[task]/` 目录中，您可以查看命令行打印的信息来获取详细位置，然后可以执行单独的Eval流程。

```bash
$ cd Code/
$ bash scripts/example_gen_recognition.sh
```

**Step2: Evaluation**

我们仍以Eval的脚本距离，在我们提供的示例脚本中，我们指定了如下参数：

* 数据集
* 模型名称
* 评估维度
* 中间文件路径，即PRED_DIR

然后，执行命令：

```bash
$ bash scripts/example_eval_recognition.sh
```

在这一步之后，您可以在 `data/results/reports/recognition/` 目录下找到对应的结果文件。


### **数据集与目录结构**

现在，我们重新整理一下数据的结构，在你运行成功一次示例脚本后，您的 `data/` 目录应当形如下面的结构：

```
data/
├── processed/              # 预处理后的数据，即我们提供的数据
│   ├── recognition/
│   ├── ...
│   └── selection/
└── results/
    ├── predictions/        # 预测结果
    └── reports/            # 报告
```

- `data/processed/`：按任务划分的已处理数据目录：
  - `continuation/`、`conversation/`、`recognition/`、`selection/` 等，每个子目录存放对应任务所需的数据集文件（为 JSON或JSONL 格式）。
- `data/results/`：评估中产生的中间/最终结果：
  - `predictions/`：模型预测输出。
  - `reports/`：评估报告。

## **添加新数据或新配置**

您可以参照如下的步骤加入您自己的数据集用于评测：

- 在 `Code/configs/` 中添加新的配置文件（通过类似的结构），并在运行脚本中指定新配置文件。
- 将您的数据集对齐我们提供的数据集格式。
