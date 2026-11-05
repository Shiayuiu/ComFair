# ComFair：大语言模型交叉公平性评测基准

**Beyond Single-Attribute Auditing: A Configurable Multi-Attribute Benchmark for Intersectional Fairness in LLMs**

[EMNLP 2026 主会论文](https://aclanthology.org/2026.emnlp-main.1875/) · [arXiv](https://arxiv.org/abs/2610.35069) · [数据集](https://huggingface.co/datasets/benchmark-lab/ComFair) · [English](README.md)

ComFair 是用于审计大语言模型交叉公平性的可配置基准，将源样本与保持任务语义的多属性增强样本配对，覆盖 **Recognition、Selection、Continuation 和 Conversation** 四类任务。生成与评估分开运行，保存的预测结果可以独立评估。

## 安装

```bash
git clone https://github.com/Shiayuiu/ComFair.git
cd ComFair
conda create -n comfair python=3.10 -y
conda activate comfair
python -m pip install -r requirements.txt
```

API 推理不需要本地 GPU，但当前模型加载入口仍会导入 PyTorch 和 Transformers。如需仅安装 CPU 版 PyTorch，请先运行 `python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`，再安装 requirements。

## 数据准备

在[数据集页面](https://huggingface.co/datasets/benchmark-lab/ComFair)接受访问条件，然后从仓库根目录登录并下载：

```bash
hf auth login
hf download benchmark-lab/ComFair --repo-type dataset --local-dir Code/data/ComFair
```

将各下载目录中的 JSONL 文件复制到对应任务目录，保留原文件名：

| 下载目录 | 任务目录 |
| --- | --- |
| `Code/data/ComFair/v3/Recog_v3/` | `Code/data/processed/recognition/` |
| `Code/data/ComFair/v3/Select_v3/` | `Code/data/processed/selection/` |
| `Code/data/ComFair/v3/Continue_v3/` | `Code/data/processed/continuation/` |
| `Code/data/ComFair/v3/Conversation_v3/` | `Code/data/processed/conversation/` |

也可以将 [`Code/configs/`](Code/configs/) 中对应配置的 `data_dir` 指向下载后的任务目录。按下文命令运行时，相对路径以 `Code/` 为基准。

## 使用

在 `Code/` 下运行任务脚本。以 Recognition 为例，先在 [`example_gen_recognition.sh`](Code/scripts/example_gen_recognition.sh) 中设置模型标识、模型名称、API 地址、凭据和数据筛选条件，再将 [`example_eval_recognition.sh`](Code/scripts/example_eval_recognition.sh) 中的 `MODEL_NAME` 和 `EVAL_DIM` 设置为相同值，然后执行：

```bash
cd Code
bash scripts/example_gen_recognition.sh
bash scripts/example_eval_recognition.sh
```

`EVAL_DIM=1d` 表示在源样本上新增一个属性，`2d` 表示新增两个属性，`all` 表示同时运行两类数据。`BASE_ATTRS` 用于选择源属性，`DEBUG_N` 可限制生成样本数，便于小规模试跑（`0` 表示处理全部匹配样本）。

各任务提供 `run_gen_<task>.py` 和 `run_eval_<task>.py` 入口，其他运行脚本位于 `Code/scripts/<task>/`。Continuation 和 Conversation 的评估还需配置 judge 模型及其 API 参数；具体参数可查看对应入口的 `--help`。

## 仓库结构

```text
Code/
├── configs/           # 数据路径、字段映射和提示词
├── scripts/           # 示例及各任务运行脚本
├── src/models/        # API 与本地 Hugging Face 模型后端
├── src/metrics/       # 公共评估工具
├── run_gen_*.py       # 生成入口
└── run_eval_*.py      # 评估入口
```

预测结果保存在 `Code/data/results/predictions/<task>/`，结构化报告保存在 `Code/data/results/reports/<task>/`，Markdown 报告保存在 `Code/outputs/<task>/`。

接入自定义数据时，遵循对应任务的 JSONL 字段与文件命名约定，在 `Code/configs/<dataset>.py` 中提供 `dataset_config`，并通过 `--dataset <dataset>` 选择配置。

## 引用

```bibtex
@inproceedings{zhang-etal-2026-beyond-single,
  title={Beyond Single-Attribute Auditing: A Configurable Multi-Attribute Benchmark for Intersectional Fairness in {LLM}s},
  author={Zhang, Zheng and Wu, Yifan and Zhou, Yuxiang and Liu, Qi and Mao, Qingyang and Li, Ning and Gao, Weibo and Shi, QiYuan and Huang, Zhenya and Yu, Baosheng},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  month=oct,
  year={2026},
  address={Budapest, Hungary},
  publisher={Association for Computational Linguistics},
  pages={42742--42762},
  doi={10.18653/2026.emnlp-main.1875},
  url={https://aclanthology.org/2026.emnlp-main.1875/}
}
```

## 许可证

本仓库采用 [CC BY 4.0](LICENSE) 许可证。
