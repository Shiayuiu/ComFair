# run_gen_recognition.py
import argparse
import os
import glob
import json
import time
import pandas as pd
from tqdm import tqdm

from src.models import load_model
from configs import get_dataset_config


# ------------------------------------------------------------
# Parse model response
# ------------------------------------------------------------
def parse_response(response: str, label_map: dict) -> int:
    response = "" if response is None else str(response).strip()
    if "1" in response:
        return label_map.get("1", -1)
    if "2" in response:
        return label_map.get("2", -1)
    return -1


# ------------------------------------------------------------
# Filename parsing
# ------------------------------------------------------------
def parse_filename(fname: str):
    """
    1d__<base>__<inject>.jsonl
    2d__<base>__<inject1>__<inject2>.jsonl
    """
    if not fname.endswith(".jsonl"):
        return None, None, None

    parts = fname[:-6].split("__")
    if len(parts) < 3:
        return None, None, None

    dim = parts[0]
    base = parts[1]
    injects = parts[2:]

    if dim not in {"1d", "2d"}:
        return None, None, None

    return dim, base, injects


# ------------------------------------------------------------
# File matching logic
# ------------------------------------------------------------
def match_files(data_dir, eval_dim, base_attrs, inject_attrs):
    matched = []
    inject_set = set(inject_attrs) if inject_attrs else None

    pattern = os.path.join(data_dir, f"{eval_dim}__*.jsonl")
    all_files = glob.glob(pattern)

    print(f"[DEBUG] scan pattern: {pattern}")
    print(f"[DEBUG] found {len(all_files)} candidate files")

    for f in all_files:
        dim, base, injects = parse_filename(os.path.basename(f))
        if dim != eval_dim:
            continue
        if base not in base_attrs:
            continue

        if eval_dim == "1d" and len(injects) != 1:
            continue
        if eval_dim == "2d" and len(injects) != 2:
            continue

        if inject_set is not None and set(injects) != inject_set:
            continue

        matched.append(f)

    return sorted(set(matched))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    print(f">>> Loading dataset config: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)

    data_dir = getattr(dataset_config, "data_dir", "data/processed/recognition/")
    print(
        f">>> eval_dim={args.eval_dim} | "
        f"base_attrs={args.base_attrs} | "
        f"inject_attrs={args.inject_attrs or 'ANY'}"
    )

    target_files = []

    if args.eval_dim in {"1d", "all"}:
        target_files += match_files(
            data_dir, "1d", args.base_attrs, args.inject_attrs
        )

    if args.eval_dim in {"2d", "all"}:
        target_files += match_files(
            data_dir, "2d", args.base_attrs, args.inject_attrs
        )

    target_files = sorted(set(target_files))
    if not target_files:
        print("❌ No matching files found.")
        return

    print(f">>> Matched {len(target_files)} files:")
    for f in target_files:
        print("  -", os.path.basename(f))

    # --------------------------------------------------------
    # Build prompts
    # --------------------------------------------------------
    samples = []
    for fp in target_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                prompt = dataset_config.prompt_template.format(
                    context=r[dataset_config.column_mapping["context"]],
                    text=r[dataset_config.column_mapping["text"]],
                )
                samples.append({
                    **r,
                    "_prompt": prompt
                })

    print(f">>> Total samples: {len(samples)}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = load_model(
        model_type=args.model_type,
        model_path_or_name=args.model_path,
        api_key=args.api_key,
        base_url=args.base_url,
        max_workers=args.max_workers,
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------
    results = []
    prompts = [s["_prompt"] for s in samples]

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Inferencing"):
        batch = prompts[i:i + args.batch_size]
        outs = model.predict_batch_with_prob(batch)

        for j, out in enumerate(outs):
            meta = samples[i + j]
            pred = parse_response(out.get("text"), dataset_config.label_map)

            results.append({
                **{k: meta[k] for k in meta if k != "_prompt"},
                "pred_label": pred,
                "raw_response": out.get("text"),
                "model_name": args.model_name,
            })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    df = pd.DataFrame(results)
    os.makedirs("data/results/predictions/recognition", exist_ok=True)

    tag = "__".join([
        args.eval_dim,
        "base-" + "_".join(args.base_attrs),
        "inj-" + "_".join(args.inject_attrs) if args.inject_attrs else "inj-any",
    ])

    out = f"data/results/predictions/recognition/{args.model_name}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.parquet"
    df.to_parquet(out, index=False)

    print(f">>> Saved {len(df)} rows to {out}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bbq_augmented")
    parser.add_argument("--eval_dim", required=True, choices=["1d", "2d", "all"])
    parser.add_argument("--base_attrs", nargs="+", required=True)
    parser.add_argument("--inject_attrs", nargs="+", default=None)

    parser.add_argument("--model_type", default="api")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--debug_n", type=int, default=0)

    args = parser.parse_args()
    main(args)
