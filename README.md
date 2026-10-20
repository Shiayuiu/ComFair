# README for Beyond Single-Attribute Auditing: A Configurable Multi-Attribute Benchmark for Intersectional Fairness in LLMs

# ComFair Library

This repository provides **ComFair**, a configurable benchmark for evaluating the intersectional fairness of large language models across different tasks. The following sections describe the code organization, environment dependencies, running methods, dataset structure, and common directory instructions to help you quickly implement evaluations.

## **Main Contents and Core Features**

* **Decoupled Pipeline**: Completely separates inference (Generation) from evaluation (Evaluation).
* **Dynamic Intersectional Query**: Supports dynamic generation of intersectional subgroups for evaluation via a query list (e.g., `['race', 'gender']`), eliminating the need for hard-coded grouping logic.
* **Comprehensive Metrics**: Through basic metrics and intersectional fairness metrics, ComFair provides a multi-dimensional score for a model.
* **Config-Driven**: Supports rapid integration of new datasets by importing user-defined configuration files, enabling flexible evaluation expansion.

## **Quick Start**

### Environment Setup

1. Clone the repository and navigate into the directory:

```bash
git clone https://github.com/Shiayuiu/Intersectional_Fairness_alpha ComFair
cd ComFair
```

2. Create a virtual environment using conda:

```bash
conda create -n ComFair python=3.10 -y
conda activate ComFair
pip install -r requirements.txt
```

If you use other environment management tools (e.g., `venv`, `pipenv`, `poetry`), install the dependencies listed in `requirements.txt` accordingly.

Note: If you only need to evaluate API-based models without deploying models locally, please ignore the CUDA-related parts in `requirements.txt`.

### Data Preparation

Our data has been open-sourced on Huggingface. You can access our data via the link below, then download and place it in the `Code/data/` directory of this repository so that our evaluator can correctly read the files.

[ComFair: A Configurable Multi-Attribute Benchmark](https://huggingface.co/datasets/benchmark-lab/Intersectional_Fairness_alpha_v2/tree/main "ComFair: Configurable Multi-Attribute Benchmark ")

Note: The data path is hard-coded in the corresponding configuration files in the `Code/configs/` folder. If you need to specify a custom data location, please modify the `data_dir` parameter in the config file.

You need to create a `processed/` directory under `Code/data/` to maintain consistency with the config files. After this step, your `data/` directory should have the following structure:

```
data/
└── processed/              # Processed data, i.e., the data we provide
    ├── recognition/
    ├── ...
    └── selection/
```

### **Running Instructions**

We evaluate your model using two steps: Generation and Evaluation. The corresponding Python files for each task are located in the `Code/` directory, named as `run_gen_[task_name].py` and `run_eval_[task_name].py`. You can evaluate directly via the command line, but we recommend using scripts for evaluation to easily specify required parameters. To help you understand the script structure, we have kept the scripts we used for model evaluation in the `Code/scripts/` directory and provided two example scripts with related instructions for your reference.

**Step 1: Generation**

In the `Code/scripts/` directory, we provide examples. In this script, we specify:

* Dataset
* Evaluation dimensions, i.e., the number of intersectional attributes we consider
* Base attribute, i.e., which attribute's intersectional fairness impact we examine, used to explicitly match dataset files
* Model type: API model
* Model name and identifier (e.g., `deepseek-chat`)
* Base_url and API_key
* Some other parameters

Run the following command, and the model will automatically read the corresponding dataset's config file, perform generation, and store the intermediate results in the `Code/data/results/predictions/[task]/` directory. You can check the information printed on the command line for the exact location, then proceed with a separate evaluation process.

```bash
$ cd Code/
$ bash scripts/example_gen_recognition.sh
```

**Step 2: Evaluation**

Continuing with the evaluation script example, in the provided example script we specify the following parameters:

* Dataset
* Model name
* Evaluation dimensions
* Intermediate file path, i.e., PRED_DIR

Then execute the command:

```bash
$ bash scripts/example_eval_recognition.sh
```

After this step, you can find the corresponding result files in the `data/results/reports/recognition/` directory.

### **Dataset and Directory Structure**

Now, let's reorganize the data structure. After successfully running an example script once, your `data/` directory should look like the following:

```
data/
├── processed/              # Preprocessed data, i.e., the data we provide
│   ├── recognition/
│   ├── ...
│   └── selection/
└── results/
    ├── predictions/        # Prediction results
    └── reports/            # Reports
```

- `data/processed/`: Directory containing processed data organized by task:
  - Subdirectories such as `continuation/`, `conversation/`, `recognition/`, `selection/`, etc., each holding dataset files required for the corresponding task (in JSON or JSONL format).
- `data/results/`: Intermediate/final results generated during evaluation:
  - `predictions/`: Model prediction outputs.
  - `reports/`: Evaluation reports.

## **Adding New Data or Configurations**

You can follow the steps below to incorporate your own dataset for evaluation:

- Add a new configuration file in `Code/configs/` (following a similar structure) and specify the new configuration file in the run script.
- Align your dataset with the format of the dataset we provide.