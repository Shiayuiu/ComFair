# run_gen_selection.py
import argparse
import os
import glob
import json
import time
import traceback
from typing import List, Optional, Tuple, Dict, Any, Iterator

import pandas as pd
from tqdm import tqdm

from src.models import load_model
from configs import get_dataset_config

import concurrent.futures


# ------------------------------------------------------------
# Response Parsing
# ------------------------------------------------------------
def parse_response(response: str, label_map: dict) -> int:
    response = "" if response is None else str(response).strip()
    if not response:
        return -1

    # 精确匹配
    if response in label_map:
        return label_map[response]

    # 模糊匹配
    for key in ["0", "1", "2"]:
        if key in response:
            return label_map.get(key, -1)
    return -1


# ------------------------------------------------------------
# Filename Parsing
# ------------------------------------------------------------
def _parse_dim_attrs_from_filename(fname: str) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """
    支持命名规范：
      1d__<base>__<inject>.jsonl
      2d__<base>__<inject1>__<inject2>.jsonl
    返回：dim, base, injects(list)
    """
    if not fname.endswith(".jsonl"):
        return None, None, None

    stem = fname[:-6]  # remove .jsonl
    parts = stem.split("__")
    if len(parts) < 3:
        return None, None, None

    dim = parts[0]
    base = parts[1]
    injects = parts[2:]

    if dim not in {"1d", "2d"}:
        return None, None, None

    return dim, base, injects


# ------------------------------------------------------------
# Core Matching Logic (base-driven + optional inject constraint)
# ------------------------------------------------------------
def match_dim_files(
    data_dir: str,
    eval_dim: str,
    base_attrs: List[str],
    inject_attrs: Optional[List[str]] = None,
) -> List[str]:
    """
    支持两种模式：
    - base-driven（inject_attrs=None）：只按 base 匹配
    - inject-driven（inject_attrs 指定）：同时约束 inject 组合
    """
    matched: List[str] = []
    inject_set = set(inject_attrs) if inject_attrs else None

    for base in base_attrs:
        if eval_dim == "1d":
            pattern = os.path.join(data_dir, f"1d__{base}__*.jsonl")
            files = glob.glob(pattern)
            for f in files:
                _, _, injects = _parse_dim_attrs_from_filename(os.path.basename(f))
                if not injects or len(injects) != 1:
                    continue
                if inject_set is None or injects[0] in inject_set:
                    matched.append(f)

        elif eval_dim == "2d":
            pattern = os.path.join(data_dir, f"2d__{base}__*__*.jsonl")
            files = glob.glob(pattern)
            for f in files:
                _, _, injects = _parse_dim_attrs_from_filename(os.path.basename(f))
                if not injects or len(injects) != 2:
                    continue
                if inject_set is None or set(injects) == inject_set:
                    matched.append(f)

    return sorted(set(matched))


def pick_target_files(
    data_dir: str,
    eval_dim: str,
    base_attrs: List[str],
    inject_attrs: Optional[List[str]] = None,
) -> List[str]:
    eval_dim = eval_dim.lower().strip()
    if eval_dim not in {"1d", "2d", "all"}:
        raise ValueError(f"Invalid eval_dim: {eval_dim}")

    matched: List[str] = []
    if eval_dim in {"1d", "all"}:
        matched += match_dim_files(data_dir, "1d", base_attrs, inject_attrs)
    if eval_dim in {"2d", "all"}:
        matched += match_dim_files(data_dir, "2d", base_attrs, inject_attrs)

    return sorted(set(matched))


# ------------------------------------------------------------
# Streaming read helper
# ------------------------------------------------------------
def iter_samples_from_files(
    target_files: List[str],
    dataset_config
) -> Iterator[Tuple[Dict[str, Any], str]]:
    """
    流式读取 + 逐条产出 (meta, prompt)
    """
    for file_path in target_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    print(f"[Warn] JSON parse failed in {file_path}: {line[:160]}...")
                    continue

                prompt = dataset_config.prompt_template.format(
                    context=record.get("context", ""),
                    question=record.get("question", ""),
                    ans0=record.get("ans0", ""),
                    ans1=record.get("ans1", ""),
                    ans2=record.get("ans2", ""),
                )

                meta = {
                    "sample_id": record.get("sample_id", ""),
                    "pair_id": record.get("pair_id", ""),
                    "pair_type": record.get("pair_type", ""),
                    "base_attr": record.get("base_attr", ""),
                    "add_attr": record.get("add_attr", ""),
                    "add_attr_value": record.get("add_attr_value", ""),
                    "label": record.get("label", -1),
                    "context_condition": record.get("context_condition", "unknown"),
                }
                yield meta, prompt


def count_lines(files: List[str]) -> int:
    total = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for _ in f:
                    total += 1
        except Exception:
            pass
    return total


# ------------------------------------------------------------
# Timeout + Retry + Split Batch
# ------------------------------------------------------------
def _call_with_timeout(model, batch_prompts: List[str], timeout_s: int) -> List[dict]:
    """
    硬超时：predict_batch_with_prob 最多等待 timeout_s 秒，避免卡死。
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(model.predict_batch_with_prob, batch_prompts)
        return fut.result(timeout=timeout_s)


def predict_batch_safe(
    model,
    batch_prompts: List[str],
    timeout_s: int,
    max_retries: int,
    min_batch_size: int = 1,
) -> List[dict]:
    """
    - 超时/异常：重试
    - 仍失败：对半拆分递归
    - 最终兜底：单条仍失败 -> {"text": ""}，保证主流程继续
    """
    if not batch_prompts:
        return []

    last_err: Optional[BaseException] = None

    for attempt in range(1, max_retries + 1):
        try:
            return _call_with_timeout(model, batch_prompts, timeout_s=timeout_s)
        except concurrent.futures.TimeoutError as e:
            last_err = e
            print(f"[Timeout] batch_size={len(batch_prompts)} | attempt {attempt}/{max_retries} | timeout_s={timeout_s}")
        except Exception as e:
            last_err = e
            print(f"[Error] batch_size={len(batch_prompts)} | attempt {attempt}/{max_retries} | {type(e).__name__}: {e}")

        time.sleep(min(2 * attempt, 6))

    if len(batch_prompts) > min_batch_size:
        mid = len(batch_prompts) // 2
        left = batch_prompts[:mid]
        right = batch_prompts[mid:]
        out_left = predict_batch_safe(model, left, timeout_s, max_retries, min_batch_size=min_batch_size)
        out_right = predict_batch_safe(model, right, timeout_s, max_retries, min_batch_size=min_batch_size)
        return out_left + out_right

    print(f"[Fallback] giving empty output for one prompt due to: {repr(last_err)}")
    return [{"text": ""}]


# ------------------------------------------------------------
# Resume helper
# ------------------------------------------------------------
def load_finished_ids(tmp_jsonl: str) -> set:
    """
    从已有 tmp_jsonl 读取已完成 sample_id，用于断点续跑。
    """
    finished = set()
    if not os.path.exists(tmp_jsonl):
        return finished

    try:
        with open(tmp_jsonl, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sid = obj.get("sample_id")
                    if sid:
                        finished.add(sid)
                except Exception:
                    continue
    except Exception as e:
        print(f"[Warn] Failed to read tmp file for resume: {e}")

    return finished


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    print(f">>> Loading config for dataset: {args.dataset}")
    try:
        dataset_config = get_dataset_config(args.dataset)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    data_dir = getattr(dataset_config, "data_dir", "data/processed/selection/")
    print(
        f">>> Scanning files | eval_dim={args.eval_dim} "
        f"| base_attrs={args.base_attrs} "
        f"| inject_attrs={args.inject_attrs} "
        f"| data_dir={data_dir}"
    )

    target_files = pick_target_files(
        data_dir=data_dir,
        eval_dim=args.eval_dim,
        base_attrs=args.base_attrs,
        inject_attrs=args.inject_attrs,
    )
    if not target_files:
        print("Error: No matching files found.")
        return

    print(f">>> Found {len(target_files)} files:")
    for f in target_files:
        print("   -", os.path.basename(f))

    # -----------------------------
    # Output paths
    # -----------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/results/predictions/selection", exist_ok=True)

    tag_parts = [args.eval_dim, "base-" + "_".join(args.base_attrs)]
    if args.inject_attrs:
        tag_parts.append("inj-" + "_".join(args.inject_attrs))
    tag = "__".join(tag_parts)

    out_path = f"data/results/predictions/selection/{args.model_name}_{tag}_{timestamp}.parquet"
    tmp_jsonl = out_path.replace(".parquet", ".tmp.jsonl")

    # ✅ 断点续跑：如果目录里已经有同模型同 tag 的 tmp 文件，优先复用最新的那个
    pattern_tmp = f"data/results/predictions/selection/{args.model_name}_{tag}_*.tmp.jsonl"
    existing_tmps = sorted(glob.glob(pattern_tmp))
    if existing_tmps:
        latest_tmp = existing_tmps[-1]
        if not os.path.exists(tmp_jsonl):
            tmp_jsonl = latest_tmp
            out_path = tmp_jsonl.replace(".tmp.jsonl", ".parquet")

    # -----------------------------
    # Resume: load finished ids
    # -----------------------------
    finished_ids = load_finished_ids(tmp_jsonl)
    if finished_ids:
        print(f">>> Detected existing tmp file, enabling resume: {tmp_jsonl}")
        print(f">>> Resume: {len(finished_ids)} samples already finished")
    else:
        print(f">>> Writing temp stream to: {tmp_jsonl}")

    # -----------------------------
    # Load Model (respect args)
    # -----------------------------
    model_kwargs: Dict[str, Any] = {}
    if args.api_key:
        model_kwargs["api_key"] = args.api_key
    if args.base_url:
        model_kwargs["base_url"] = args.base_url

    env_force = os.getenv("SELECTION_API_MAX_WORKERS")
    if env_force and env_force.isdigit():
        effective_workers = int(env_force)
    else:
        effective_workers = int(args.max_workers)

    model_kwargs["max_workers"] = effective_workers
    print(f">>> Model max_workers: {effective_workers} (args.max_workers={args.max_workers})")

    model = load_model(
        model_type=args.model_type,
        model_path_or_name=args.model_path,
        **model_kwargs,
    )

    # -----------------------------
    # Runtime config
    # -----------------------------
    timeout_s = int(os.getenv("SELECTION_TIMEOUT_S", "180"))
    max_retries = int(os.getenv("SELECTION_RETRY", "4"))
    checkpoint_every = int(os.getenv("SELECTION_CHECKPOINT_EVERY", "50"))
    print(">>> Runtime config:")
    print(f"    - batch_size        : {args.batch_size}")
    print(f"    - timeout_s (batch) : {timeout_s}")
    print(f"    - max_retries       : {max_retries}")
    print(f"    - checkpoint_every  : {checkpoint_every}")

    # -----------------------------
    # Progress total
    # -----------------------------
    total_lines = count_lines(target_files)
    print(f">>> Total raw lines (approx samples): {total_lines}")

    if int(args.debug_n) > 0:
        print(f">>> DEBUG MODE enabled: debug_n={args.debug_n} (limits newly written samples)")

    # -----------------------------
    # Streaming inference loop
    # -----------------------------
    written_new = 0
    skipped_existing = 0
    fallback_empty = 0

    buffer_meta: List[Dict[str, Any]] = []
    buffer_prompts: List[str] = []

    def flush_buffer(fout):
        nonlocal written_new, fallback_empty, buffer_meta, buffer_prompts

        if not buffer_prompts:
            return

        batch_output = predict_batch_safe(
            model,
            buffer_prompts,
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_batch_size=1,
        )

        # 对齐长度
        if len(batch_output) != len(buffer_prompts):
            print(f"[Warn] batch_output size mismatch: {len(batch_output)} vs {len(buffer_prompts)}; aligning...")
            if len(batch_output) < len(buffer_prompts):
                batch_output = batch_output + [{"text": ""}] * (len(buffer_prompts) - len(batch_output))
            else:
                batch_output = batch_output[: len(buffer_prompts)]

        for meta, out in zip(buffer_meta, batch_output):
            raw_text = out.get("text", "") if isinstance(out, dict) else str(out)
            if not str(raw_text).strip():
                fallback_empty += 1

            pred_label = parse_response(raw_text, dataset_config.label_map)

            row = {
                **meta,
                "pred_label": pred_label,
                "raw_response": raw_text,
                "model_name": args.model_name,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written_new += 1

        buffer_meta = []
        buffer_prompts = []

    # 进度条：已完成的也算进度（更直观）
    pbar = tqdm(total=total_lines if total_lines > 0 else None, desc="Inferencing", unit="sample")

    try:
        # ✅ 追加写，确保续跑不覆盖
        with open(tmp_jsonl, "a", encoding="utf-8") as fout:
            # debug_n：限制“新写入”的样本数（跳过已完成的不算）
            debug_limit = int(args.debug_n) if int(args.debug_n) > 0 else None

            for meta, prompt in iter_samples_from_files(target_files, dataset_config):
                # 达到 debug 限制就停止（先不 flush，下面会处理尾巴）
                if debug_limit is not None and written_new >= debug_limit:
                    break

                sid = meta.get("sample_id", "")
                if sid and sid in finished_ids:
                    skipped_existing += 1
                    pbar.update(1)
                    continue

                buffer_meta.append(meta)
                buffer_prompts.append(prompt)

                # 如果 debug 剩余不足一个 batch，就动态缩小 batch
                if debug_limit is not None:
                    remaining = debug_limit - written_new
                    target_bs = min(int(args.batch_size), max(1, remaining))
                else:
                    target_bs = int(args.batch_size)

                if len(buffer_prompts) >= target_bs:
                    flush_buffer(fout)
                    pbar.update(target_bs)

                    if written_new % checkpoint_every == 0:
                        fout.flush()
                        os.fsync(fout.fileno())

            # flush 尾巴（也要尊重 debug_limit）
            if buffer_prompts:
                if debug_limit is not None:
                    remaining = debug_limit - written_new
                    if remaining <= 0:
                        buffer_meta = []
                        buffer_prompts = []
                    elif len(buffer_prompts) > remaining:
                        buffer_meta = buffer_meta[:remaining]
                        buffer_prompts = buffer_prompts[:remaining]

                before = len(buffer_prompts)
                if before > 0:
                    flush_buffer(fout)
                    pbar.update(before)

    except KeyboardInterrupt:
        print("\n[Interrupted] Ctrl+C pressed. tmp preserved for resume:", tmp_jsonl)
        return
    except Exception as e:
        print("[Fatal] Unexpected error:", e)
        traceback.print_exc()
        print("[Note] tmp preserved for resume:", tmp_jsonl)
        return
    finally:
        pbar.close()

    print(">>> Resume summary:")
    print(f"    - skipped_existing (already done): {skipped_existing}")
    print(f"    - newly_written (this run):       {written_new}")
    if written_new > 0:
        print(f"    - empty/fallback responses:       {fallback_empty} ({fallback_empty / written_new * 100:.2f}%)")

    # -----------------------------
    # Convert tmp -> parquet (dedup by sample_id)
    # -----------------------------
    try:
        df = pd.read_json(tmp_jsonl, lines=True)
        if "sample_id" in df.columns:
            df = df.drop_duplicates(subset=["sample_id"], keep="last")

        df.to_parquet(out_path, index=False)
        print(f">>> Saved predictions to {out_path}")
        print(f">>> Total rows in parquet: {len(df)}")

    except Exception as e:
        print("[Error] Failed to convert to parquet:", e)
        print("[Note] tmp preserved:", tmp_jsonl)
        return


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bbq_selection")
    parser.add_argument("--eval_dim", type=str, choices=["1d", "2d", "all"], required=True)

    parser.add_argument("--base_attrs", nargs="+", required=True)
    parser.add_argument("--inject_attrs", nargs="+", default=None)

    parser.add_argument("--model_type", type=str, default="hf")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--debug_n", type=int, default=0)  # ✅ 现在生效：限制“新写入”的样本数

    args = parser.parse_args()
    main(args)
