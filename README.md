# ComFair: Intersectional Fairness Benchmark for LLMs

**Beyond Single-Attribute Auditing: A Configurable Multi-Attribute Benchmark for Intersectional Fairness in LLMs**

[EMNLP 2026 · Main Conference](https://aclanthology.org/2026.emnlp-main.1875/) · [arXiv](https://arxiv.org/abs/2610.35069) · [Dataset](https://huggingface.co/datasets/benchmark-lab/ComFair) · [中文文档](README_zh_CN.md)

ComFair is a configurable benchmark for auditing intersectional fairness in large language models. It pairs source instances with task-preserving, multi-attribute counterparts across four tasks: **Recognition, Selection, Continuation, and Conversation**. Generation and evaluation run separately, so saved predictions can be evaluated independently.

## Installation

```bash
git clone https://github.com/Shiayuiu/ComFair.git
cd ComFair
conda create -n comfair python=3.10 -y
conda activate comfair
python -m pip install -r requirements.txt
```

API inference does not require a local GPU, but the current model loader still imports PyTorch and Transformers. For a CPU-only setup, run `python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu` before installing the requirements.

## Data preparation

Accept the access conditions on the [dataset page](https://huggingface.co/datasets/benchmark-lab/ComFair), then authenticate and download from the repository root:

```bash
hf auth login
hf download benchmark-lab/ComFair --repo-type dataset --local-dir Code/data/ComFair
```

Copy the JSONL files from each downloaded folder into its task directory, preserving filenames:

| Downloaded folder | Task directory |
| --- | --- |
| `Code/data/ComFair/v3/Recog_v3/` | `Code/data/processed/recognition/` |
| `Code/data/ComFair/v3/Select_v3/` | `Code/data/processed/selection/` |
| `Code/data/ComFair/v3/Continue_v3/` | `Code/data/processed/continuation/` |
| `Code/data/ComFair/v3/Conversation_v3/` | `Code/data/processed/conversation/` |

Alternatively, set `data_dir` in the relevant file under [`Code/configs/`](Code/configs/) to the downloaded task folder. Relative paths are resolved from `Code/` when running the commands below.

## Usage

Run task scripts from `Code/`. For a Recognition example, edit [`example_gen_recognition.sh`](Code/scripts/example_gen_recognition.sh) to set the model identifier, model name, API endpoint, credentials, and data filters. Set the same `MODEL_NAME` and `EVAL_DIM` in [`example_eval_recognition.sh`](Code/scripts/example_eval_recognition.sh), then run:

```bash
cd Code
bash scripts/example_gen_recognition.sh
bash scripts/example_eval_recognition.sh
```

`EVAL_DIM=1d` adds one attribute to the base instance; `2d` adds two; `all` includes both. Use `BASE_ATTRS` to select the source attribute and `DEBUG_N` to limit generation for a small trial (`0` processes all matched samples).

Each task has `run_gen_<task>.py` and `run_eval_<task>.py` entry points. Additional scripts are in `Code/scripts/<task>/`. Continuation and Conversation evaluation also require a judge model and its API configuration; see each entry point's `--help` for arguments.

## Repository structure

```text
Code/
├── configs/           # Dataset paths, field mappings, and prompts
├── scripts/           # Example and task-specific launch scripts
├── src/models/        # API and local Hugging Face model backends
├── src/metrics/       # Shared evaluation utilities
├── run_gen_*.py       # Generation entry points
└── run_eval_*.py      # Evaluation entry points
```

Predictions are saved to `Code/data/results/predictions/<task>/`, structured reports to `Code/data/results/reports/<task>/`, and Markdown reports to `Code/outputs/<task>/`.

To use a custom dataset, follow the corresponding task's JSONL schema and filename conventions, add `Code/configs/<dataset>.py` exposing `dataset_config`, and select it with `--dataset <dataset>`.

## Citation

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

## License

This repository is distributed under [CC BY 4.0](LICENSE).
