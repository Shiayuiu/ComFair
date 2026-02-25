# run_gen_classification.py
import re
import argparse
import pandas as pd
import os
import time
from configs import get_dataset_config
from src.models import load_model
from tqdm import tqdm


def parse_response_backup(response_text: str, label_map: dict) -> int:
    """
    Simple parser that maps model output to a binary label (0 or 1).

    Returns -1 if parsing fails.
    """
    clean_text = response_text.strip().upper().replace(".", "")

    # Exact match first
    if clean_text in label_map:
        return label_map[clean_text]

    # Fuzzy match (e.g., "Option A", "Result: A")
    for key, val in label_map.items():
        if key in clean_text:
            return val

    return -1


def parse_response(response_text: str, label_map: dict) -> int:
    """
    Robust parser that prioritizes matching the leading token
    to avoid accidental matches inside explanation text.
    """
    if not response_text:
        return -1

    text = response_text.strip().upper()

    # Strategy 1: match the first token
    first_token = text.split()[0]
    clean_token = re.sub(r"[^\w]", "", first_token)

    if clean_token in label_map:
        return label_map[clean_token]

    # Strategy 2: regex-based boundary match (e.g., "Answer: A")
    for key, val in label_map.items():
        pattern = fr"(?:^|\s|:){key}(?:\.|,|$|\s)"
        if re.search(pattern, text):
            return val

    return -1


def main(args):
    # ------------------------------------------------------------
    # 1) Load dataset configuration and prepare output path
    # ------------------------------------------------------------
    print(f"Loading configuration for dataset: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)

    output_dir = "data/results/predictions/classification/"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(
        output_dir,
        f"{args.model_name}-{args.dataset}-{timestamp}.parquet"
    )

    # ------------------------------------------------------------
    # 2) Load processed feature file
    # ------------------------------------------------------------
    print(f"Loading features from {dataset_config.processed_feature_path}...")
    df = pd.read_csv(dataset_config.processed_feature_path)

    # Debug mode: limit to first N samples
    if args.debug_n > 0:
        df = df.head(args.debug_n)

    # ------------------------------------------------------------
    # 3) Construct prompts dynamically
    # ------------------------------------------------------------
    print("Constructing prompts...")
    prompts = []

    for _, row in df.iterrows():
        format_dict = {
            std_key: row[csv_col]
            for std_key, csv_col in dataset_config.column_mapping.items()
        }
        prompt = dataset_config.prompt_template.format(**format_dict)
        prompts.append(prompt)

    # ------------------------------------------------------------
    # 4) Load model
    # ------------------------------------------------------------
    model_kwargs = {}
    if args.api_key:
        model_kwargs["api_key"] = args.api_key
    if args.base_url:
        model_kwargs["base_url"] = args.base_url

    model = load_model(
        model_type=args.model_type,
        model_path_or_name=args.model_path,
        **model_kwargs
    )

    # ------------------------------------------------------------
    # 5) Batched inference
    # ------------------------------------------------------------
    print(f"Running inference on {len(prompts)} samples...")
    batch_size = args.batch_size

    all_raw_texts = []
    all_preds = []
    all_probs = []

    pbar = tqdm(total=len(prompts), desc="Inference", unit="sample")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        batch_results = model.predict_batch_with_prob(batch_prompts)

        for res in batch_results:
            text = res["text"]
            prob = res["probability"]

            pred_label = parse_response(text, dataset_config.label_map)

            all_raw_texts.append(text)
            all_preds.append(pred_label)
            all_probs.append(prob)

        pbar.update(len(batch_results))

    pbar.close()

    # ------------------------------------------------------------
    # 6) Aggregate and save results
    # ------------------------------------------------------------
    result_df = pd.DataFrame({
        "sample_id": df["sample_id"],
        "prompt": prompts,
        "raw_response": all_raw_texts,
        "pred_label": all_preds,
        # "pred_prob": all_probs,
        "model_name": args.model_name
    })

    result_df.to_parquet(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-1.5B", help="Model identifier")
    parser.add_argument("--model_path", type=str, required=True, help="Local path or HuggingFace hub path")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--debug_n", type=int, default=0, help="Run only N samples for debugging")

    parser.add_argument(
        "--model_type",
        type=str,
        default="hf",
        choices=["hf", "api", "reasoning_api"],
        help="Backend type: local HuggingFace or API-based model"
    )

    parser.add_argument("--api_key", type=str, default=None, help="API key for API-based models")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for API-based models")

    args = parser.parse_args()
    main(args)