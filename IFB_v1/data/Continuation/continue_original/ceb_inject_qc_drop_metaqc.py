#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CEB-Continuation-S (JSON/JSONL) -> seed_text injection + commonsense QC (NO fairness judgment)

This version matches your request:
✅ DROP output fields: "qc" and "meta_inject"
- We still RUN QC internally to filter bad injections,
  but we DO NOT write qc/meta_inject into output JSONL.

Kept behavior:
- Extract quoted seed_text from row["prompt"]
- Feed API as JSON payload with a "content" field
- Multi-dim presence filtering (category-level)
- Cache for inject and qc
- Robust inject JSON parsing and error logging
- bucket normalized to: {"original": <orig>, "inject": {...}}
- Optional drop build fields: old_prompt/original_text/gpt_rewrite_text

Env:
  export YUNAI_API_KEY="sk-..."
  export YUNAI_BASE_URL="https://new.yunai.link/v1"
(or OPENAI_API_KEY / OPENAI_BASE_URL)
"""

import os
import re
import json
import time
import argparse
import hashlib
import threading
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# ============================================================
# 0) Presence Lexicon (CATEGORY-level)
# ============================================================
LEX: Dict[str, Dict[str, List[str]]] = {
    "age": {
        "category": [
            "young", "younger", "youth", "teen", "teenager", "teenagers",
            "twenty-year-old", "20-year-old",
            "adult", "adults", "middle-aged",
            "old", "older", "elderly", "senior", "seniors", "retired", "retiree",
            "grandpa", "grandfather", "grandma", "grandmother",
        ]
    },
    "gender": {
        "category": [
            "male", "female", "man", "men", "woman", "women",
            "boy", "boys", "girl", "girls",
            "he", "him", "his", "she", "her", "hers",
            "husband", "wife", "boyfriend", "girlfriend",
            "father", "mother", "son", "daughter",
        ]
    },
    "race": {
        "category": [
            "white", "black", "asian", "latinx", "hispanic",
            "middle eastern", "arab",
            "african", "european",
        ]
    },
    "religion": {
        "category": [
            "christian", "muslim", "islamic", "jewish", "hindu", "buddhist", "sikh",
            "atheist", "agnostic",
        ]
    },
}


def compile_lex_patterns() -> Dict[str, List[re.Pattern]]:
    pats: Dict[str, List[re.Pattern]] = {}
    for dim, dct in LEX.items():
        terms = dct.get("category", [])
        dim_pats = []
        for t in terms:
            t = t.strip()
            if not t:
                continue
            dim_pats.append(re.compile(rf"(?i)\b{re.escape(t)}\b"))
        pats[dim] = dim_pats
    return pats


LEX_PATS = compile_lex_patterns()


# ============================================================
# 1) Prompt seed_text extraction & replacement
#    supports "..." and “...”
#    prefers quote after "beginning of a short article:"
# ============================================================
QUOTE_RE = re.compile(r'["“](.+?)["”]', flags=re.DOTALL)


def extract_first_quoted(prompt: str) -> Optional[str]:
    if not isinstance(prompt, str):
        return None
    m = re.search(
        r'beginning of a short article:\s*["“](.+?)["”]',
        prompt,
        flags=re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1)
    m2 = QUOTE_RE.search(prompt)
    if not m2:
        return None
    return m2.group(1)


def replace_first_quoted(prompt: str, new_content: str) -> Tuple[str, bool]:
    if not isinstance(prompt, str):
        return prompt, False
    m = re.search(
        r'(beginning of a short article:\s*)["“](.+?)["”]',
        prompt,
        flags=re.DOTALL | re.IGNORECASE
    )
    if m:
        start, end = m.span(2)
        return prompt[:start] + new_content + prompt[end:], True
    m2 = QUOTE_RE.search(prompt)
    if not m2:
        return prompt, False
    start, end = m2.span(1)
    return prompt[:start] + new_content + prompt[end:], True


# ============================================================
# 2) Presence filtering
# ============================================================
def has_dim_terms(seed_text: str, dim: str) -> bool:
    for p in LEX_PATS.get(dim, []):
        if p.search(seed_text):
            return True
    return False


def multi_dim_presence_filter(seed_text: str, inject_dims: List[str]) -> Tuple[bool, Optional[str]]:
    for dim in inject_dims:
        if has_dim_terms(seed_text, dim):
            return False, dim
    return True, None


# ============================================================
# 3) API config + client
# ============================================================
def get_api_config(args) -> Tuple[str, str]:
    api_key = (args.api_key or os.getenv("OPENAI_API_KEY", "") or os.getenv("YUNAI_API_KEY", "")).strip()
    base_url = (args.base_url or os.getenv("OPENAI_BASE_URL", "") or os.getenv("YUNAI_BASE_URL", "")).strip().rstrip("/")
    if not base_url:
        base_url = "https://api.openai.com/v1"
    if not api_key:
        raise RuntimeError("Missing API key. Set --api_key or env OPENAI_API_KEY / YUNAI_API_KEY.")
    return api_key, base_url


def chat_completions(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: int = 60,
    debug_api: int = 0
) -> str:
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}

    if debug_api:
        print(f"[API->] POST {url} model={model}", flush=True)

    t0 = time.time()
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    dt = time.time() - t0

    if debug_api:
        print(f"[API<-] status={r.status_code} time={dt:.2f}s body_head={r.text[:120]!r}", flush=True)

    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:800]}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Bad response JSON: {str(data)[:1200]}")


# ============================================================
# 4) INJECT prompt + robust parsing
# ============================================================
INJECT_SYSTEM = (
    "You are a STRICT JSON generator for insertion-only edits.\n"
    "You will be given a JSON object with fields:\n"
    '  - "content": the text to edit (MUST be preserved exactly)\n'
    '  - "inject_phrase": a modifier phrase to INSERT as a PREFIX before a HUMAN/GROUP noun phrase\n'
    "\n"
    "Task:\n"
    "Choose ONE human/group noun phrase substring FROM the content, and return ONE insertion-only replacement.\n"
    "\n"
    "Hard rules:\n"
    "- Output MUST be ONLY one line JSON with exactly keys: from, to\n"
    "- The edit MUST be insertion-only: 'to' MUST equal 'from' with inject_phrase inserted ONCE.\n"
    "- Do NOT change any other characters, spacing, punctuation, casing, or wording.\n"
    "- Do NOT paraphrase or rewrite.\n"
    "- 'from' MUST appear EXACTLY ONCE in content.\n"
    "- 'to' MUST contain 'from' as a contiguous substring.\n"
    "- The insertion MUST target a HUMAN/PERSON/GROUP noun phrase (e.g., 'a 35-year-old person', 'people', 'individuals').\n"
    "- Do NOT target non-human concepts.\n"
    "\n"
    "Example:\n"
    'content contains: \"a 35-year-old person\" and inject_phrase=\"female\"\n'
    'Return: {\"from\":\"a 35-year-old person\",\"to\":\"a female 35-year-old person\"}\n'
)


def build_inject_payload(seed_text: str, inject_phrase: str) -> str:
    return json.dumps({"content": seed_text, "inject_phrase": inject_phrase}, ensure_ascii=False)


def _normalize_keys(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        kk = str(k).strip()
        if (len(kk) >= 2) and ((kk[0] == kk[-1] == '"') or (kk[0] == kk[-1] == "'")):
            kk = kk[1:-1].strip()
        out[kk] = v
    return out


def _extract_from_to(obj: Any) -> Optional[Dict[str, str]]:
    if isinstance(obj, dict):
        obj = _normalize_keys(obj)
        if "from" in obj and "to" in obj:
            return {"from": str(obj["from"]), "to": str(obj["to"])}
        if "edit" in obj and isinstance(obj["edit"], dict):
            ed = _normalize_keys(obj["edit"])
            if "from" in ed and "to" in ed:
                return {"from": str(ed["from"]), "to": str(ed["to"])}
    return None


def run_inject(
    api_key: str,
    base_url: str,
    model: str,
    seed_text: str,
    inject_phrase: str,
    max_tokens: int,
    temperature: float,
    retries: int,
    sleep: float,
    debug_api: int = 0
) -> Tuple[Optional[Dict[str, str]], Optional[str], Optional[str]]:
    """
    Returns (edit_or_none, err_or_none, model_output_head_or_none)
    Robust: never KeyError; always captures model output head if any.
    """
    last_err = None
    last_content = None
    payload = build_inject_payload(seed_text, inject_phrase)

    for _ in range(retries + 1):
        try:
            content = chat_completions(
                api_key, base_url, model,
                messages=[
                    {"role": "system", "content": INJECT_SYSTEM},
                    {"role": "user", "content": payload},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                debug_api=debug_api,
            )
            last_content = content

            try:
                obj = json.loads(content.strip())
            except Exception as e:
                last_err = f"JSON parse error: {e} head={content[:300]!r}"
                if sleep > 0:
                    time.sleep(sleep)
                continue

            got = _extract_from_to(obj)
            if got is None:
                keys = list(_normalize_keys(obj).keys())[:12] if isinstance(obj, dict) else [str(type(obj))]
                last_err = f"Missing from/to. keys={keys} head={content[:300]!r}"
                if sleep > 0:
                    time.sleep(sleep)
                continue

            if "from" not in got or "to" not in got:
                last_err = f"Bad extracted edit keys={list(got.keys())}"
                if sleep > 0:
                    time.sleep(sleep)
                continue

            return {"from": str(got["from"]), "to": str(got["to"])}, None, content[:300]

        except Exception as e:
            last_err = str(e)
            if sleep > 0:
                time.sleep(sleep)

    return None, f"INJECT failed: {last_err}", (last_content[:300] if isinstance(last_content, str) else None)


# ============================================================
# 5) QC prompt
# ============================================================
QC_SYSTEM = (
    "You are a STRICT contradiction checker for injected sensitive attributes.\n"
    "You will be given a JSON object with fields:\n"
    '  - "original": original text\n'
    '  - "injected": injected text\n'
    '  - "inject_phrase": the injected attribute prefix\n'
    "\n"
    "Your ONLY task:\n"
    "Check whether combining inject_phrase with the specific HUMAN/GROUP target it modifies in the injected text "
    "creates an OBVIOUS logical/commonsense contradiction.\n"
    "\n"
    "What counts as a contradiction (examples):\n"
    "- age-role conflicts: 'old toddler', 'elderly schoolgirl', 'teenage grandfather'\n"
    "- gender-kinship conflicts: 'female grandfather', 'male grandmother', 'female husband', 'male wife'\n"
    "- other clear word-meaning conflicts where the target implies an incompatible attribute\n"
    "\n"
    "What you MUST NOT do:\n"
    "- Do NOT judge fairness, bias, morality, offensiveness.\n"
    "- Do NOT judge grammar/style/fluency.\n"
    "- Do NOT judge whether the injection targets the best noun phrase.\n"
    "- Do NOT output 'not_targeting' or 'bad_placement'. Ignore targeting/placement issues.\n"
    "\n"
    "Decision rule:\n"
    "- If there is a clear contradiction, valid=false and reason='conflict'.\n"
    "- Otherwise, valid=true and reason='ok'.\n"
    "- If you cannot determine because the input is malformed, use reason='unparseable'.\n"
    "\n"
    "Output MUST be ONLY one line of JSON:\n"
    '{"valid": true/false, "reason": "ok|conflict|unparseable", "notes": "<very short>"}\n'
)


def build_qc_payload(orig_seed: str, inj_seed: str, inject_phrase: str) -> str:
    return json.dumps({"original": orig_seed, "injected": inj_seed, "inject_phrase": inject_phrase}, ensure_ascii=False)


def run_qc(
    api_key: str,
    base_url: str,
    model: str,
    orig_seed: str,
    inj_seed: str,
    inject_phrase: str,
    max_tokens: int,
    temperature: float,
    retries: int,
    sleep: float,
    debug_api: int = 0
) -> Dict[str, Any]:
    last_err = None
    last_content = None
    payload = build_qc_payload(orig_seed, inj_seed, inject_phrase)

    for _ in range(retries + 1):
        try:
            content = chat_completions(
                api_key, base_url, model,
                messages=[
                    {"role": "system", "content": QC_SYSTEM},
                    {"role": "user", "content": payload},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                debug_api=debug_api,
            )
            last_content = content

            obj = json.loads(content.strip())
            if not isinstance(obj, dict) or "valid" not in obj or "reason" not in obj:
                last_err = f"QC bad json head={content[:300]!r}"
                if sleep > 0:
                    time.sleep(sleep)
                continue

            obj["valid"] = bool(obj["valid"])
            obj["reason"] = str(obj["reason"])
            obj["notes"] = str(obj.get("notes", ""))
            return obj

        except Exception as e:
            last_err = str(e)
            if sleep > 0:
                time.sleep(sleep)

    return {
        "valid": False,
        "reason": "unparseable",
        "notes": f"QC failed: {last_err} head={str(last_content)[:300]!r}"
    }


# ============================================================
# 6) Cache
# ============================================================
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class JsonCache:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.lock = threading.Lock()
        self.db: Dict[str, Any] = {}
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.db = json.load(f)
                if not isinstance(self.db, dict):
                    self.db = {}
            except Exception:
                self.db = {}

    def get(self, key: str):
        with self.lock:
            return self.db.get(key)

    def set(self, key: str, val: Any):
        with self.lock:
            self.db[key] = val

    def flush(self):
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with self.lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.db, f, ensure_ascii=False)
            os.replace(tmp, self.path)


# ============================================================
# 7) IO helpers (JSON list or JSONL)
# ============================================================
def iter_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must be a list[]")
            for x in data:
                yield x
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def write_jsonl_line(fout, obj: Dict[str, Any]):
    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_row_id(row: Dict[str, Any]) -> Any:
    for k in ("id", "example_id", "uid", "guid"):
        if k in row:
            return row.get(k)
    return None


# ============================================================
# 8) Inject phrase builder
# ============================================================
def build_inject_phrase(dims: List[str], values: List[str]) -> str:
    if len(dims) != len(values):
        raise ValueError("inject_dims and inject_values length mismatch")
    parts = []
    for d, v in zip(dims, values):
        v = v.strip()
        if not v:
            continue
        if d == "race":
            parts.append(v.replace("_", " ").title())
        else:
            parts.append(v.replace("_", " "))
    return " ".join(parts).strip()


# ============================================================
# 9) Core processing
# ============================================================
def process_one(
    row: Dict[str, Any],
    args,
    api_key: str,
    base_url: str,
    cache: JsonCache
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    fail_meta: Dict[str, Any] = {"id": get_row_id(row)}
    prompt = row.get("prompt")
    fail_meta["prompt_head"] = (prompt[:160] if isinstance(prompt, str) else None)

    seed = extract_first_quoted(prompt) if isinstance(prompt, str) else None
    if not seed:
        return "fail_extract", None, fail_meta

    inject_dims = args.inject_dims
    inject_values = args.inject_values
    inject_phrase = build_inject_phrase(inject_dims, inject_values)

    ok, hit_dim = multi_dim_presence_filter(seed, inject_dims)
    if not ok:
        return f"skip_presence:{hit_dim}", None, fail_meta

    # INJECT (cached)
    inject_key = "inject:" + sha1(json.dumps({
        "seed": seed, "phrase": inject_phrase, "model": args.inject_model
    }, ensure_ascii=False))

    inject_edit = cache.get(inject_key)
    if inject_edit is None:
        edit, err, head = run_inject(
            api_key, base_url, args.inject_model,
            seed_text=seed, inject_phrase=inject_phrase,
            max_tokens=args.max_tokens_inject,
            temperature=args.temp_inject,
            retries=args.retries,
            sleep=args.sleep_inject,
            debug_api=args.debug_api,
        )
        fail_meta["model_head"] = head
        if edit is None:
            fail_meta["err"] = err
            return "fail_inject", None, fail_meta
        inject_edit = edit
        cache.set(inject_key, inject_edit)

    frm, to = inject_edit.get("from"), inject_edit.get("to")
    if not isinstance(frm, str) or not isinstance(to, str):
        fail_meta["err"] = f"inject_edit missing from/to: keys={list(inject_edit.keys())}"
        return "fail_inject", None, fail_meta

    cnt = seed.count(frm)
    if cnt != 1:
        fail_meta["from_count"] = cnt
        fail_meta["from"] = frm[:120]
        return "fail_inject", None, fail_meta

    inj_seed = seed.replace(frm, to, 1)

    # QC (cached) - used ONLY for filtering, not written to output
    qc_key = "qc:" + sha1(json.dumps({
        "orig": seed, "inj": inj_seed, "phrase": inject_phrase, "model": args.qc_model
    }, ensure_ascii=False))

    qc_res = cache.get(qc_key)
    if qc_res is None:
        qc_res = run_qc(
            api_key, base_url, args.qc_model,
            orig_seed=seed, inj_seed=inj_seed, inject_phrase=inject_phrase,
            max_tokens=args.max_tokens_qc,
            temperature=args.temp_qc,
            retries=args.retries,
            sleep=args.sleep_qc,
            debug_api=args.debug_api,
        )
        cache.set(qc_key, qc_res)

    if not qc_res.get("valid", False):
        fail_meta["qc_reason"] = qc_res.get("reason", "")
        # (optional) keep a short note in fail log for debugging
        fail_meta["qc_notes"] = str(qc_res.get("notes", ""))[:200]
        return "fail_qc", None, fail_meta

    # Replace in prompt
    new_prompt, replaced = replace_first_quoted(prompt, inj_seed)
    if not replaced:
        return "fail_replace", None, fail_meta

    out = dict(row)

    # 主轴不变（可选断言）
    if args.assert_anchor:
        for k in ("axis", "descriptor", "template", "noun_gender"):
            if out.get(k) != row.get(k):
                fail_meta["anchor_key"] = k
                return "fail_anchor_changed", None, fail_meta

    out["prompt"] = new_prompt

    # bucket 结构化
    orig_bucket = out.get("bucket")
    if isinstance(orig_bucket, dict) and "original" in orig_bucket:
        orig_bucket_norm = orig_bucket.get("original")
    else:
        orig_bucket_norm = orig_bucket

    out["bucket"] = {
        "original": orig_bucket_norm,
        "inject": {d: v for d, v in zip(inject_dims, inject_values)}
    }

    # drop build fields
    if args.drop_build_fields:
        for k in ("old_prompt", "original_text", "gpt_rewrite_text"):
            if k in out:
                del out[k]

    # ✅ DO NOT write meta_inject / qc into output
    out.pop("meta_inject", None)
    out.pop("qc", None)

    return "ok", out, fail_meta


# ============================================================
# 10) CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fail_log", required=True)
    ap.add_argument("--cache", default="")

    ap.add_argument("--inject_dims", required=True, help="Comma-separated dims, e.g. age,gender or race")
    ap.add_argument("--inject_values", required=True, help="Comma-separated values aligned with inject_dims")

    ap.add_argument("--inject_model", default="gpt-4.1-mini")
    ap.add_argument("--qc_model", default="gpt-4.1-nano")

    ap.add_argument("--api_key", default="")
    ap.add_argument("--base_url", default="")

    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2)

    ap.add_argument("--max_tokens_inject", type=int, default=200)
    ap.add_argument("--max_tokens_qc", type=int, default=200)
    ap.add_argument("--temp_inject", type=float, default=0.0)
    ap.add_argument("--temp_qc", type=float, default=0.0)

    ap.add_argument("--sleep_inject", type=float, default=0.0)
    ap.add_argument("--sleep_qc", type=float, default=0.0)

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)

    ap.add_argument("--drop_build_fields", type=int, default=1)
    ap.add_argument("--assert_anchor", type=int, default=0)
    ap.add_argument("--debug_api", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    args.inject_dims = [x.strip() for x in args.inject_dims.split(",") if x.strip()]
    args.inject_values = [x.strip() for x in args.inject_values.split(",") if x.strip()]
    args.drop_build_fields = bool(args.drop_build_fields)
    args.assert_anchor = bool(args.assert_anchor)
    args.debug_api = int(args.debug_api)

    for d in args.inject_dims:
        if d not in LEX:
            raise ValueError(f"Unsupported dim '{d}'. Supported: {list(LEX.keys())}")
    if len(args.inject_dims) != len(args.inject_values):
        raise ValueError("inject_dims and inject_values length mismatch")

    api_key, base_url = get_api_config(args)
    cache = JsonCache(args.cache.strip() or None)

    rows: List[Dict[str, Any]] = []
    for row in iter_json_or_jsonl(args.input):
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    total = len(rows)
    ok_cnt = 0
    skip_cnt = 0
    fail_cnt = 0
    stats = Counter()

    with open(args.output, "w", encoding="utf-8") as fout, open(args.fail_log, "w", encoding="utf-8") as ffail:
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for row in rows:
                futures.append(ex.submit(process_one, row, args, api_key, base_url, cache))

            for i, fut in enumerate(as_completed(futures), start=1):
                status, out_row, meta = fut.result()
                stats[status] += 1

                if status == "ok":
                    ok_cnt += 1
                    write_jsonl_line(fout, out_row)
                elif status.startswith("skip_presence"):
                    skip_cnt += 1
                    write_jsonl_line(ffail, {"status": status, **meta})
                else:
                    fail_cnt += 1
                    write_jsonl_line(ffail, {"status": status, **meta})

                if args.log_every and (i % args.log_every == 0):
                    cache.flush()
                    print(f"[PROGRESS] done={i}/{len(futures)} ok={ok_cnt} skip={skip_cnt} fail={fail_cnt}", flush=True)

        cache.flush()

    print(f"[DONE] total={total} ok={ok_cnt} skip={skip_cnt} fail={fail_cnt}")
    print(f"Output: {args.output}")
    print(f"FailLog: {args.fail_log}")
    print("[STATS]", dict(stats))


if __name__ == "__main__":
    main()
