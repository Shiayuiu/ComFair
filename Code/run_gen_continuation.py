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
    新命名协议：
      1d__<base>__<inject>.jsonl
      2d__<base>__<inject1>__<inject2>.jsonl
    返回 (dim, base, injects) 或 None
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
    """允许用户传入 'gender,race' 这种写法，统一拆开成 ['gender','race']"""
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
    逐行读取 JSONL，并生成 prompt，yield meta dict（包含 _prompt_for_inf）。
    这样不会一次性把所有样本塞进内存，避免“假卡住”。
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
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def safe_text(x) -> str:
    if x is None:
        return "[EMPTY]"
    x = str(x)
    if not x.strip():
        return "[EMPTY]"
    return x


def run_predict_batch_with_timeout(model, prompts: List[str], max_new_tokens: int, timeout_s: int):
    """
    给 predict_batch 增加超时。
    注意：这依赖 ThreadPoolExecutor，无法强制杀死底层卡住的网络调用，
    但可以让主流程“继续往下跑”，避免整个任务挂死。
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
    # 1) 加载配置
    print(f">>> Loading config for dataset: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)

    # 兼容旧用法：base_attrs 默认等于 attributes
    if args.base_attrs is None:
        args.base_attrs = args.attributes

    # inject_attrs 规范化
    args.inject_attrs = normalize_inject_attrs(args.inject_attrs)

    # 2) 扫描文件（新命名优先；不匹配则 fallback 旧命名）
    target_files = []
    data_dir = getattr(dataset_config, "data_dir", "data/processed/continuation/")
    print(f">>> Scanning files in {data_dir}...")

    all_files = glob.glob(os.path.join(data_dir, "*.jsonl"))

    for f in all_files:
        fname = os.path.basename(f)
        parsed = parse_dim_file(fname)

        if parsed is None:
            # fallback：师兄旧逻辑（attr_*.jsonl）
            if fname.startswith(tuple([a + "_" for a in args.attributes])):
                target_files.append(f)
            continue

        dim, base, injects = parsed

        # 维度过滤
        if args.eval_dim != "all" and dim != args.eval_dim:
            continue

        # 主轴过滤
        if base not in args.base_attrs:
            continue

        # 注入过滤（可选，集合精确匹配，顺序无关）
        if args.inject_attrs:
            if set(injects) != set(args.inject_attrs):
                continue

        target_files.append(f)

    target_files = sorted(list(set(target_files)))
    if not target_files:
        print("Error: No matching files found.")
        print(">>> Tips:")
        print("  - New naming expects: 1d__<base>__<inj>.jsonl or 2d__<base>__<inj1>__<inj2>.jsonl")
        print("  - Or fallback old naming: <attr>_*.jsonl where attr in --attributes")
        return

    print(f">>> Found {len(target_files)} files.")

    # 3) 加载目标模型
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
        model_path_or_name=args.model_path,  # continuation 脚本就用 model_path 参数承载“模型名”
        **model_kwargs,
    )
    print(">>> Model loaded.")

    # 4) 逐批生成（边读边推理，避免一次性读全量）
    record_iter = iter_records_from_files(target_files, dataset_config)

    # debug_n：只取前 N 条
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

    # ✅ 改动：文件名带上 eval_dim（由 bash 的 --eval_dim 控制）
    dim_tag = getattr(args, "eval_dim", "unk")
    out_path = f"data/results/predictions/continuation/{args.model_name}__{dim_tag}__{timestamp}.parquet"

    results: List[Dict] = []
    total_batches_est = None  # 流式读取无法精确知道总 batch，这里用 tqdm 的不定长模式

    pbar = tqdm(desc="Generating", unit="batch", total=total_batches_est)
    batch_id = 0

    for batch in batched(record_iter, args.batch_size):
        batch_id += 1
        prompts = [b["_prompt_for_inf"] for b in batch]

        # 关键日志：让你知道确实开始请求了
        if args.verbose:
            print(f">>> [BATCH {batch_id}] calling API | size={len(prompts)} | max_new_tokens={args.max_new_tokens}")

        status, outs, err = run_predict_batch_with_timeout(
            model=model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            timeout_s=args.request_timeout,
        )

        if status == "ok":
            # outs 应该是 list[str]
            if not isinstance(outs, list):
                # 某些 wrapper 可能返回单个字符串
                outs = [outs] * len(batch)

            # 对齐长度
            if len(outs) != len(batch):
                # 不要直接崩，填充
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
            # 超时：本 batch 全部标记 timeout
            for meta in batch:
                meta2 = dict(meta)
                meta2["raw_response"] = "[TIMEOUT]"
                meta2["model_name"] = args.model_name
                meta2["status"] = "timeout"
                meta2["error"] = err
                meta2.pop("_prompt_for_inf", None)
                results.append(meta2)

        else:  # error
            # 出错：本 batch 全部标记 error
            # 这里把 traceback 也记录一份，方便你定位（但不会刷屏）
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

        # 每个 batch 都落一次盘（可选），避免中途挂了全没
        if args.flush_every > 0 and (batch_id % args.flush_every == 0):
            df = pd.DataFrame(results)
            df.to_parquet(out_path, index=False)
            if args.verbose:
                print(f">>> [FLUSH] saved {len(results)} rows to {out_path}")

        pbar.update(1)

    pbar.close()

    # 最终落盘
    df = pd.DataFrame(results)
    df.to_parquet(out_path, index=False)
    print(f">>> Generation saved to {out_path}")
    print(">>> Status counts:")
    try:
        print(df["status"].value_counts(dropna=False).to_string())
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="holistic_continuation")

    # 旧参数（保持 required=True，兼容你现有 bash）
    parser.add_argument("--attributes", nargs="+", required=True)

    # 新增参数（跟 conversation 一样）
    parser.add_argument("--eval_dim", default="1d", choices=["1d", "2d", "all"])
    parser.add_argument("--base_attrs", nargs="+", default=None)
    parser.add_argument("--inject_attrs", nargs="+", default=None)

    parser.add_argument("--model_type", type=str, default="hf")
    parser.add_argument("--model_name", type=str, required=True)

    # 这里依旧叫 model_path：在 api 模式下它承载“模型名”，在 hf 模式下就是路径
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--debug_n", type=int, default=0)

    # continuation 特有
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Length of continuation")

    # 新增：稳住“卡住”的关键参数
    parser.add_argument("--request_timeout", type=int, default=180, help="Timeout seconds per batch API call")
    parser.add_argument("--flush_every", type=int, default=5, help="Flush to parquet every N batches; 0 disables")
    parser.add_argument("--verbose", action="store_true", help="Print per-batch logs")

    args = parser.parse_args()
    main(args)
