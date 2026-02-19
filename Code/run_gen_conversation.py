# run_gen_conversation.py
import argparse
import os
import glob
import json
import time
import pandas as pd
from tqdm import tqdm

from src.models import load_model
from configs import get_dataset_config


def parse_dim_file(fname: str):
    """
    支持新命名：
      1d__<base>__<inject>.jsonl
      2d__<base>__<inject1>__<inject2>.jsonl

    返回：(dim, base, injects) 或 None
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


def _safe_filename_part(s: str) -> str:
    """
    将字符串变成安全文件名片段：保留字母数字.-，其他替换为 _，
    并压缩连续下划线。
    """
    s = "" if s is None else str(s)
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ".-":
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    return out if out else "na"


def main(args):
    print(f">>> Loading config for dataset: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)

    # 兼容旧用法：base_attrs 默认等于 attributes
    if args.base_attrs is None:
        args.base_attrs = args.attributes

    # 允许用户写成 inject_attrs 里带逗号（比如 "gender,race"），这里做一次友好处理
    if args.inject_attrs is not None:
        fixed = []
        for x in args.inject_attrs:
            x = "" if x is None else str(x)
            if "," in x:
                fixed.extend([p.strip() for p in x.split(",") if p.strip()])
            else:
                fixed.append(x.strip())
        args.inject_attrs = [x for x in fixed if x]

    # ------------------------------------------------------------
    # 1) 扫描文件（优先新命名；新命名不匹配则 fallback 旧命名）
    # ------------------------------------------------------------
    target_files = []
    data_dir = getattr(dataset_config, "data_dir", "data/processed/conversation/")
    print(f">>> Scanning files in {data_dir}...")

    all_files = glob.glob(os.path.join(data_dir, "*.jsonl"))

    for f in all_files:
        fname = os.path.basename(f)
        parsed = parse_dim_file(fname)

        if parsed is None:
            # fallback：旧 1D 命名规则（不破坏师兄逻辑）
            # 原逻辑：for attr in args.attributes -> glob(f"{attr}_*.jsonl")
            if fname.startswith(tuple([a + "_" for a in args.attributes])):
                target_files.append(f)
            continue

        dim, base, injects = parsed

        # 维度过滤
        if args.eval_dim != "all" and dim != args.eval_dim:
            continue

        # 主轴过滤（base_attr）
        if base not in args.base_attrs:
            continue

        # 注入属性过滤（可选，精确匹配集合）
        if args.inject_attrs:
            if set(injects) != set(args.inject_attrs):
                continue

        target_files.append(f)

    if not target_files:
        print("❌ Error: No matching files found.")
        print(">>> Tips:")
        print("  - Check data_dir:", data_dir)
        print("  - New naming expects: 1d__<base>__<inj>.jsonl or 2d__<base>__<inj1>__<inj2>.jsonl")
        print("  - Or fallback old naming: <attr>_*.jsonl where attr in --attributes")
        return

    target_files = sorted(list(set(target_files)))
    print(f">>> Found {len(target_files)} files.")

    # ------------------------------------------------------------
    # 2) 构建 Prompts
    # ------------------------------------------------------------
    all_samples = []
    for file_path in target_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                prompt = dataset_config.gen_prompt_template.format(text=record.get("text", ""))
                meta = record.copy()
                meta["_prompt_for_inf"] = prompt
                all_samples.append(meta)

    print(f">>> Total samples to generate: {len(all_samples)}")

    # ------------------------------------------------------------
    # 3) 加载模型
    # ------------------------------------------------------------
    model_kwargs = {}
    if args.api_key:
        model_kwargs["api_key"] = args.api_key
    if args.base_url:
        model_kwargs["base_url"] = args.base_url
    if args.max_workers:
        model_kwargs["max_workers"] = args.max_workers

    model_path_or_name = getattr(args, "model_path_or_name", None) or args.model_path

    model = load_model(
        model_type=args.model_type,
        model_path_or_name=model_path_or_name,
        **model_kwargs,
    )

    # ------------------------------------------------------------
    # 4) 批量生成 (带进度条)
    # ------------------------------------------------------------
    prompts = [s["_prompt_for_inf"] for s in all_samples]
    if args.debug_n > 0:
        prompts = prompts[: args.debug_n]
        all_samples = all_samples[: args.debug_n]

    results = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Response", unit="batch"):
        batch_prompts = prompts[i : i + batch_size]
        batch_output = model.predict_batch(batch_prompts, max_new_tokens=args.max_new_tokens)

        for j, text_out in enumerate(batch_output):
            sample_idx = i + j
            meta = all_samples[sample_idx]
            results.append(
                {
                    **meta,
                    "raw_response": text_out,
                    "model_name": args.model_name,
                }
            )
            # 清理内部字段
            results[-1].pop("_prompt_for_inf", None)

    # ------------------------------------------------------------
    # 5) 保存结果（✅ 文件名：不再出现 noinject；没 inject 就省略 inject 段）
    # ------------------------------------------------------------
    df = pd.DataFrame(results)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = "data/results/predictions/conversation"
    os.makedirs(out_dir, exist_ok=True)

    model_tag = _safe_filename_part(args.model_name)
    dim_tag = _safe_filename_part(args.eval_dim)

    base_list = args.base_attrs if args.base_attrs is not None else args.attributes
    base_tag = "-".join(_safe_filename_part(x) for x in (base_list or [])) or "base"

    inject_list = args.inject_attrs or []

    if inject_list:
        inject_tag = "-".join(_safe_filename_part(x) for x in inject_list)
        out_name = f"{model_tag}__{dim_tag}__{base_tag}__{inject_tag}__{timestamp}.parquet"
    else:
        # ✅ 没有 inject_attrs，就不写 inject 段
        out_name = f"{model_tag}__{dim_tag}__{base_tag}__{timestamp}.parquet"

    out_path = os.path.join(out_dir, out_name)
    df.to_parquet(out_path, index=False)
    print(f">>> Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="holistic_conversation")

    # 旧参数：保持 required=True（不破坏你现有 bash）
    parser.add_argument("--attributes", nargs="+", required=True)

    # 新增参数：维度与过滤
    parser.add_argument("--eval_dim", default="1d", choices=["1d", "2d", "all"])
    parser.add_argument("--base_attrs", nargs="+", default=None)
    parser.add_argument("--inject_attrs", nargs="+", default=None)

    # 模型参数
    parser.add_argument("--model_type", type=str, default="hf")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    # API / 推理参数
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--debug_n", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=150)

    args = parser.parse_args()
    main(args)
