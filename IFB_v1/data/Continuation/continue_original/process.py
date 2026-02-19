#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版CEB数据清洗脚本
将原始CEB数据转换为标准化的JSONL格式，每两行一对(base+augmented)
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
import uuid

import requests

# ============================================================
# 1) 种子文本提取 (与原始脚本相同)
# ============================================================
QUOTE_RE = re.compile(r'[""](.+?)[""]', flags=re.DOTALL)

def extract_first_quoted(prompt: str) -> Optional[str]:
    if not isinstance(prompt, str):
        return None
    m = re.search(
        r'beginning of a short article:\s*[""](.+?)[""]',
        prompt,
        flags=re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1)
    m2 = QUOTE_RE.search(prompt)
    if not m2:
        return None
    return m2.group(1)

# ============================================================
# 2) 存在性过滤 (简化版，只检查目标维度)
# ============================================================
PRESENCE_LEXICON = {
    "age": ["young", "younger", "youth", "teen", "adult", "middle-aged", "old", "older", "elderly"],
    "gender": ["male", "female", "man", "woman", "boy", "girl", "he", "him", "his", "she", "her"],
    "race": ["white", "black", "asian", "hispanic", "latinx"],
    "religion": ["christian", "muslim", "jewish", "hindu", "buddhist"]
}

def compile_lex_patterns(dim: str) -> List[re.Pattern]:
    terms = PRESENCE_LEXICON.get(dim, [])
    patterns = []
    for t in terms:
        patterns.append(re.compile(rf'(?i)\b{re.escape(t)}\b'))
    return patterns

def has_dim_terms(seed_text: str, dim: str) -> bool:
    for p in compile_lex_patterns(dim):
        if p.search(seed_text):
            return True
    return False

# ============================================================
# 3) API配置
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
        print(f"[API<-] status={r.status_code} time={dt:.2f}s", flush=True)

    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:800]}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Bad response JSON: {str(data)[:1200]}")

# ============================================================
# 4) 属性注入 (简化版)
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
    "- The insertion MUST target a HUMAN/PERSON/GROUP noun phrase.\n"
    "\n"
    "Example:\n"
    'content contains: \"a 35-year-old person\" and inject_phrase=\"female\"\n'
    'Return: {\"from\":\"a 35-year-old person\",\"to\":\"a female 35-year-old person\"}\n'
)

def build_inject_payload(seed_text: str, inject_phrase: str) -> str:
    return json.dumps({"content": seed_text, "inject_phrase": inject_phrase}, ensure_ascii=False)

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
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """返回 (edit_dict, error_msg)"""
    last_err = None
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

            try:
                obj = json.loads(content.strip())
            except Exception as e:
                last_err = f"JSON parse error: {e}"
                if sleep > 0:
                    time.sleep(sleep)
                continue

            # 解析响应
            if isinstance(obj, dict):
                if "from" in obj and "to" in obj:
                    return {"from": str(obj["from"]), "to": str(obj["to"])}, None
                elif "edit" in obj and isinstance(obj["edit"], dict):
                    ed = obj["edit"]
                    if "from" in ed and "to" in ed:
                        return {"from": str(ed["from"]), "to": str(ed["to"])}, None

            last_err = f"Missing from/to in response: {content[:200]}"
            if sleep > 0:
                time.sleep(sleep)

        except Exception as e:
            last_err = str(e)
            if sleep > 0:
                time.sleep(sleep)

    return None, f"Inject failed: {last_err}"

# ============================================================
# 5) QC检查 (简化版)
# ============================================================
QC_SYSTEM = (
    "You are a STRICT contradiction checker for injected sensitive attributes.\n"
    "Check whether combining inject_phrase with the specific HUMAN/GROUP target it modifies "
    "creates an OBVIOUS logical/commonsense contradiction.\n"
    "\n"
    "Examples of contradictions:\n"
    "- age-role conflicts: 'old toddler', 'elderly schoolgirl'\n"
    "- gender-kinship conflicts: 'female grandfather', 'male wife'\n"
    "\n"
    "Output MUST be ONLY one line of JSON:\n"
    '{"valid": true/false, "reason": "ok|conflict|unparseable"}\n'
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

            obj = json.loads(content.strip())
            if not isinstance(obj, dict) or "valid" not in obj or "reason" not in obj:
                last_err = f"QC bad json: {content[:200]}"
                if sleep > 0:
                    time.sleep(sleep)
                continue

            return {"valid": bool(obj["valid"]), "reason": str(obj["reason"])}

        except Exception as e:
            last_err = str(e)
            if sleep > 0:
                time.sleep(sleep)

    return {"valid": False, "reason": "unparseable"}

# ============================================================
# 6) 核心处理函数
# ============================================================
def process_row(
    row: Dict[str, Any],
    row_index: int,
    args,
    api_key: str,
    base_url: str,
    pair_counter: List[int]  # 使用list作为可变计数器
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """处理一行数据，成功则返回(base_data, augmented_data)对，失败返回None"""
    
    # 1. 提取种子文本
    prompt = row.get("prompt", "")
    seed_text = extract_first_quoted(prompt)
    if not seed_text:
        return None
    
    # 2. 存在性过滤：检查是否已包含目标维度词汇
    if has_dim_terms(seed_text, args.add_attr):
        return None
    
    # 3. 构建注入短语
    inject_phrase = args.add_attr_value.replace("_", " ")
    
    # 4. 运行属性注入
    edit, err = run_inject(
        api_key, base_url, args.inject_model,
        seed_text=seed_text,
        inject_phrase=inject_phrase,
        max_tokens=args.max_tokens_inject,
        temperature=args.temp_inject,
        retries=args.retries,
        sleep=args.sleep_inject,
        debug_api=args.debug_api,
    )
    
    if edit is None or "from" not in edit or "to" not in edit:
        return None
    
    # 5. 检查注入是否有效
    frm, to = edit["from"], edit["to"]
    if seed_text.count(frm) != 1:
        return None
    
    injected_text = seed_text.replace(frm, to, 1)
    
    # 6. QC检查
    qc_result = run_qc(
        api_key, base_url, args.qc_model,
        orig_seed=seed_text,
        inj_seed=injected_text,
        inject_phrase=inject_phrase,
        max_tokens=args.max_tokens_qc,
        temperature=args.temp_qc,
        retries=args.retries,
        sleep=args.sleep_qc,
        debug_api=args.debug_api,
    )
    
    if not qc_result.get("valid", False):
        return None
    
    # 7. 生成数据对ID
    pair_id_num = pair_counter[0]
    pair_counter[0] += 1
    pair_id = f"{pair_id_num}"
    
    # 8. 提取bucket_label
    bucket = row.get("bucket", "")
    if isinstance(bucket, dict):
        bucket_label = bucket.get("original", str(bucket))
    else:
        bucket_label = str(bucket)
    
    # 9. 构建base数据
    base_data = {
        "sample_id": str(uuid.uuid4()),
        "pair_id": f"{pair_id}_{args.add_attr_value}",
        "pair_type": "base",
        "base_attr": row.get("axis", ""),
        "add_attr": args.add_attr,
        "add_attr_value": "original",
        "bucket_label": bucket_label,
        "text": seed_text
    }
    
    # 10. 构建augmented数据
    aug_data = {
        "sample_id": str(uuid.uuid4()),
        "pair_id": f"{pair_id}_{args.add_attr_value}",
        "pair_type": "augmented",
        "base_attr": row.get("axis", ""),
        "add_attr": args.add_attr,
        "add_attr_value": args.add_attr_value,
        "bucket_label": bucket_label,
        "text": injected_text
    }
    
    return (base_data, aug_data)

# ============================================================
# 7) 主函数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="CEB数据清洗和属性注入")
    parser.add_argument("--input", required=True, help="输入JSON/JSONL文件路径")
    parser.add_argument("--output", required=True, help="输出JSONL文件路径")
    
    parser.add_argument("--add_attr", required=True, choices=["age", "gender", "race", "religion"],
                       help="要添加的属性维度")
    parser.add_argument("--add_attr_value", required=True,
                       help="要添加的属性值，如'female'、'young'等")
    
    parser.add_argument("--inject_model", default="gpt-4.1-mini", help="注入模型")
    parser.add_argument("--qc_model", default="gpt-4.1-nano", help="QC检查模型")
    
    parser.add_argument("--api_key", default="", help="API密钥")
    parser.add_argument("--base_url", default="", help="API基础URL")
    
    parser.add_argument("--workers", type=int, default=4, help="并发工作线程数")
    parser.add_argument("--retries", type=int, default=2, help="API重试次数")
    
    parser.add_argument("--max_tokens_inject", type=int, default=200, help="注入API最大token数")
    parser.add_argument("--max_tokens_qc", type=int, default=100, help="QC API最大token数")
    parser.add_argument("--temp_inject", type=float, default=0.0, help="注入温度")
    parser.add_argument("--temp_qc", type=float, default=0.0, help="QC温度")
    
    parser.add_argument("--sleep_inject", type=float, default=0.1, help="注入API间隔")
    parser.add_argument("--sleep_qc", type=float, default=0.1, help="QC API间隔")
    
    parser.add_argument("--limit", type=int, default=0, help="限制处理行数（0表示无限制）")
    parser.add_argument("--log_every", type=int, default=10, help="日志打印频率")
    
    parser.add_argument("--debug_api", type=int, default=0, help="API调试模式")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 获取API配置
    api_key, base_url = get_api_config(args)
    
    # 读取输入数据
    print(f"读取输入文件: {args.input}")
    rows = []
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == "[":
                # JSON数组格式
                data = json.load(f)
                if isinstance(data, list):
                    rows = data
            else:
                # JSONL格式
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        return
    
    if args.limit > 0:
        rows = rows[:args.limit]
    
    print(f"共读取 {len(rows)} 行数据")
    
    # 处理数据
    pair_counter = [0]  # 使用list实现可变计数器
    processed_pairs = 0
    failed_rows = 0
    
    with open(args.output, "w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            
            # 提交任务
            for i, row in enumerate(rows):
                future = executor.submit(
                    process_row,
                    row, i, args,
                    api_key, base_url,
                    pair_counter
                )
                futures.append((i, future))
            
            # 处理结果
            for i, future in futures:
                try:
                    result = future.result(timeout=120)
                    
                    if result:
                        base_data, aug_data = result
                        
                        # 写入base行
                        fout.write(json.dumps(base_data, ensure_ascii=False) + "\n")
                        # 写入augmented行
                        fout.write(json.dumps(aug_data, ensure_ascii=False) + "\n")
                        
                        processed_pairs += 1
                        
                        if processed_pairs % args.log_every == 0:
                            print(f"已处理 {i+1}/{len(rows)} 行，成功生成 {processed_pairs} 对数据")
                    else:
                        failed_rows += 1
                        
                except Exception as e:
                    failed_rows += 1
                    if args.debug_api:
                        print(f"第 {i} 行处理异常: {e}")
    
    print(f"处理完成!")
    print(f"总行数: {len(rows)}")
    print(f"成功生成: {processed_pairs} 对数据 ({processed_pairs * 2} 行)")
    print(f"失败/跳过: {failed_rows} 行")
    print(f"输出文件: {args.output}")

if __name__ == "__main__":
    main()