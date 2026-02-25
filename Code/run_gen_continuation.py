# run_gen_continuation.py
import argparse
import os
import glob
import json
import time
import traceback
from typing import Dict, List, Optional, Iterator

import pandas as pd
from tqdm import tqdm

from src.models import load_model
from configs import get_dataset_config


def parse_dim_file(fname: str):
    """
    Supported naming convention:
      1d__<base>__<inject>.jsonl
      2d__<base>__<inject1>__<inject2>.jsonl

    Returns (dim, base, injects) or None.
    """
    if not fname.endswith(".jsonl"):
        return None
    stem = fname[:-6]  # remove .jsonl
    parts = stem.split("__")
    if len(parts) < 3:
        return None
    dim = parts[0]
    base = parts[1]
    injects = parts[2:]
    if dim not in {"1d", "2d"}:
        return None
    if not base:
        return None
    if any(x.strip() == "" for x in injects):
        return None
    return dim, base, injects


def normalize_inject_attrs(inject_attrs):
    """
    Normalize inject attributes.

    Allows inputs such as "gender,race" and converts them into
    ['gender', 'race'].
    """
    if inject_attrs is None:
        return None
    fixed = []
    for x in inject_attrs:
        if x is None:
            continue
        x = str(x).strip()
        if not x:
            continue
        if "," in x:
            fixed.extend([p.strip() for p in x.split(",") if p.strip()])
        else:
            fixed.append(x)
    return fixed if fixed else None


def iter_records_from_files(
    files: List[str],
    dataset_config,
) -> Iterator[Dict]:
    """
    Stream JSONL files line by line and yield meta dictionaries
    containing generated prompts.

    This avoids loading all samples into memory at once.
    """
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                prompt = dataset_config.gen_prompt_template.format(text=record.get("text", ""))
                meta = record.copy()
                meta["_prompt_for_inf"] = prompt
                meta["_source_file"] = os.path.basename(file_path)
                yield meta


def batched(iterator: Iterator[Dict], batch_size: int) -> Iterator[List[Dict]]:
    """
    Yield data in fixed-size batches.
    """
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def safe_text(x) -> str:
    """
    Ensure non-empty string output.
    """
    if x is None:
        return "[EMPTY]"
    x = str(x)
    if not x.strip():
        return "[EMPTY]"
    return x


def run_predict_batch_with_timeout(model, prompts: List[str], max_new_tokens: int, timeout_s: int):
    """
    Wrap predict_batch with a timeout mechanism.

    Note:
    This relies on ThreadPoolExecutor and cannot forcibly terminate
    a blocked underlying network call. However, it allows the main
    process to continue instead of hanging indefinitely.
    """
    import concurrent.futures

    def _call():
        return model.predict_batch(prompts, max_new_tokens=max_new_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            return "ok", fut.result(timeout=timeout_s), None
        except concurrent.futures.TimeoutError:
            return "timeout", None, f"Timeout after {timeout_s}s"
        except Exception as e:
            return "error", None, f"{type(e).__name__}: {e}"


def main(args):
    # ------------------------------------------------------------
    # 1) Load dataset configuration
    # ------------------------------------------------------------
    print(f">>> Loading config for dataset: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)

    # Backward compatibility: base_attrs defaults to attributes
    if args.base_attrs is None:
        args.base_attrs = args.attributes

    # Normalize inject attributes
    args.inject_attrs = normalize_inject_attrs(args.inject_attrs)

    # ------------------------------------------------------------
    # 2) Scan data files (new naming preferred; fallback to legacy)
    # ------------------------------------------------------------
    target_files = []
    data_dir = getattr(dataset_config, "data_dir", "data/processed/continuation/")
    print(f">>> Scanning files in {data_dir}...")

    all_files = glob.glob(os.path.join(data_dir, "*.jsonl"))

    for f in all_files:
        fname = os.path.basename(f)
        parsed = parse_dim_file(fname)

        if parsed is None:
            # Fallback to legacy naming: <attr>_*.jsonl
            if fname.startswith(tuple([a + "_" for a in args.attributes])):
                target_files.append(f)
            continue

        dim, base, injects = parsed

        # Dimension filtering
        if args.eval_dim != "all" and dim != args.eval_dim:
            continue

        # Base attribute filtering
        if base not in args.base_attrs:
            continue

        # Inject attribute filtering (set equality, order-invariant)
        if args.inject_attrs:
            if set(injects) != set(args.inject_attrs):
                continue

        target_files.append(f)

    target_files = sorted(list(set(target_files)))
    if not target_files:
        print("Error: No matching files found.")
        print(">>> Tips:")
        print("  - Expected new naming: 1d__<base>__<inj>.jsonl or 2d__<base>__<inj1>__<inj2>.jsonl")
        print("  - Legacy naming: <attr>_*.jsonl where attr in --attributes")
        return

    print(f">>> Found {len(target_files)} files.")

    # ------------------------------------------------------------
    # 3) Load target model
    # ------------------------------------------------------------
    model_kwargs = {}
    if args.api_key:
        model_kwargs["api_key"] = args.api_key
    if args.base_url:
        model_kwargs["base_url"] = args.base_url
    if args.max_workers:
        model_kwargs["max_workers"] = args.max_workers

    print(">>> Loading model...")
    model = load_model(
        model_type=args.model_type,
        model_path_or_name=args.model_path,
        **model_kwargs,
    )
    print(">>> Model loaded.")

    # ------------------------------------------------------------
    # 4) Stream inference (read and generate incrementally)
    # ------------------------------------------------------------
    record_iter = iter_records_from_files(target_files, dataset_config)

    # Limit number of samples in debug mode
    if args.debug_n and args.debug_n > 0:
        def _limited(it, n):
            c = 0
            for x in it:
                yield x
                c += 1
                if c >= n:
                    break
        record_iter = _limited(record_iter, args.debug_n)

    os.makedirs("data/results/predictions/continuation", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    dim_tag = getattr(args, "eval_dim", "unk")
    out_path = f"data/results/predictions/continuation/{args.model_name}__{dim_tag}__{timestamp}.parquet"

    results: List[Dict] = []

    pbar = tqdm(desc="Generating", unit="batch")
    batch_id = 0

    for batch in batched(record_iter, args.batch_size):
        batch_id += 1
        prompts = [b["_prompt_for_inf"] for b in batch]

        status, outs, err = run_predict_batch_with_timeout(
            model=model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            timeout_s=args.request_timeout,
        )

        if status == "ok":
            if not isinstance(outs, list):
                outs = [outs] * len(batch)

            if len(outs) != len(batch):
                fixed = []
                for i in range(len(batch)):
                    fixed.append(outs[i] if i < len(outs) else "[MISSING_OUTPUT]")
                outs = fixed

            for meta, text_out in zip(batch, outs):
                meta2 = dict(meta)
                meta2["raw_response"] = safe_text(text_out)
                meta2["model_name"] = args.model_name
                meta2["status"] = "ok"
                meta2.pop("_prompt_for_inf", None)
                results.append(meta2)

        elif status == "timeout":
            for meta in batch:
                meta2 = dict(meta)
                meta2["raw_response"] = "[TIMEOUT]"
                meta2["model_name"] = args.model_name
                meta2["status"] = "timeout"
                meta2["error"] = err
                meta2.pop("_prompt_for_inf", None)
                results.append(meta2)

        else:
            tb = traceback.format_exc(limit=5)
            for meta in batch:
                meta2 = dict(meta)
                meta2["raw_response"] = "[ERROR]"
                meta2["model_name"] = args.model_name
                meta2["status"] = "error"
                meta2["error"] = err
                meta2["traceback"] = tb
                meta2.pop("_prompt_for_inf", None)
                results.append(meta2)

        # Periodic flushing to avoid total data loss
        if args.flush_every > 0 and (batch_id % args.flush_every == 0):
            df = pd.DataFrame(results)
            df.to_parquet(out_path, index=False)

        pbar.update(1)

    pbar.close()

    # Final save
    df = pd.DataFrame(results)
    df.to_parquet(out_path, index=False)
    print(f">>> Generation saved to {out_path}")
    print(">>> Status counts:")
    try:
        print(df["status"].value_counts(dropna=False).to_string())
    except Exception:
        pass
    
# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="holistic_continuation")

    parser.add_argument("--attributes", nargs="+", required=True)

    parser.add_argument("--eval_dim", default="1d", choices=["1d", "2d", "all"])
    parser.add_argument("--base_attrs", nargs="+", default=None)
    parser.add_argument("--inject_attrs", nargs="+", default=None)

    parser.add_argument("--model_type", type=str, default="hf")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--debug_n", type=int, default=0)

    parser.add_argument("--max_new_tokens", type=int, default=500, help="Length of continuation")

    parser.add_argument("--request_timeout", type=int, default=180, help="Timeout seconds per batch API call")
    parser.add_argument("--flush_every", type=int, default=5, help="Flush to parquet every N batches; 0 disables")
    parser.add_argument("--verbose", action="store_true", help="Print per-batch logs")

    args = parser.parse_args()
    main(args)