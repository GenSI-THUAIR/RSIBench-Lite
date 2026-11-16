#!/usr/bin/env python3
"""从官方 LongBench v2 data.json 按上下文档位冻结正确性 gate 子集。

参数:
    data_json   官方 THUDM/LongBench-v2 的 data.json(503 题)
    out_dir     输出目录;写出每档一个 jsonl 与一份 index json

分档规则(与 verify/text/longbench_eval.py 的官方 0-shot 模板同源渲染 prompt,
再用本机 HDFS 挂载上的三份 tokenizer——Qwen3.8-27B / GLM-5.2 / LLaDA2.1——
分别计数,取最大值作为该题的 token 长度):

    档位   prompt 上限(token)          适用 serve 上下文
    24k    24576 × 0.75 = 18432         LLaDA2.1-mini(model max_input 24576)
    64k    65536 × 0.75 = 49152         64k 上下文的题
    128k   131072 × 0.75 = 98304        128k~133k 上下文的题
    256k   262144 × 0.75 = 196608       256k 及以上上下文的题

每档取"能装下的最长 100 题"(24k 档全量只有 36 题能装下,则取全部),
25% 余量留给其他模型 tokenizer 差异与输出预算(max_tokens)。选中的题按
seed 42 打乱后写出,使 smoke 口径取前 N 题时长度分布有代表性。

输出 jsonl 字段:index/_id/domain/sub_domain/difficulty/length/question/
choices{A-D}/answer/context/prompt_tokens_max;index json 记录每档 _id 列表、
题数、token 统计与 sha256,入库 datasets/gate/ 供离线核对(jsonl 本体
过大只放 HDFS,见 assets/manifest.yaml)。

示例:
    .venv/bin/python scripts/prepare/build_longbench_gate_data.py \
        /tmp/longbench-v2/data.json /tmp/longbench-v2/out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from verify.resources import payload_dir
sys.path.insert(0, str(payload_dir()))
import longbench_eval

VERSION = "v1"
SEED = 42
PER_TIER = 100
PROMPT_RATIO = 0.75
TIERS = {"24k": 24576, "64k": 65536, "128k": 131072, "256k": 262144}
MODEL_ROOT = "/mnt/models"
TOKENIZERS = {
    "qwen3p8": f"{MODEL_ROOT}/qwen3p8_27b_fp8/tokenizer.json",
    "glm52": f"{MODEL_ROOT}/glm52_ascend/tokenizer.json",
    "llada21": f"{MODEL_ROOT}/llada21_16b_bf16/tokenizer.json",
}
_tokenizers: dict[str, object] = {}


def to_sample(item: dict[str, object]) -> dict[str, object]:
    """官方 data.json 一条记录 → gate jsonl 记录(choice_X 收拢为 choices)。"""
    return {
        "_id": item["_id"],
        "domain": item["domain"],
        "sub_domain": item["sub_domain"],
        "difficulty": item["difficulty"],
        "length": item["length"],
        "question": item["question"],
        "choices": {key: item[f"choice_{key}"] for key in "ABCD"},
        "answer": item["answer"],
        "context": item["context"],
    }


def count_tokens(args: tuple[str, str]) -> int:
    """用指定 tokenizer 计一条 prompt 的 token 数(子进程内缓存 tokenizer)。"""
    name, prompt = args
    from tokenizers import Tokenizer

    if name not in _tokenizers:
        _tokenizers[name] = Tokenizer.from_file(TOKENIZERS[name])
    return len(_tokenizers[name].encode(prompt, add_special_tokens=False).ids)  # type: ignore[attr-defined]


def prompt_tokens_max(samples: list[dict[str, object]]) -> list[int]:
    """每题按官方模板渲染后,三份 tokenizer 计数取最大值。"""
    prompts = [longbench_eval.format_prompt(sample) for sample in samples]
    result = [0] * len(prompts)
    with ProcessPoolExecutor(max_workers=12) as pool:
        for name in TOKENIZERS:
            counts = pool.map(count_tokens, [(name, p) for p in prompts], chunksize=4)
            result = [max(a, b) for a, b in zip(result, counts)]
            print(f"tokenized with {name}", flush=True)
    return result


def select_tier(samples: list[dict[str, object]], cap: int) -> list[dict[str, object]]:
    """取 prompt_tokens_max ≤ cap 的最长 PER_TIER 题(不足则全取),seed 打乱。"""
    fit = [s for s in samples if int(s["prompt_tokens_max"]) <= cap]
    fit.sort(key=lambda s: (-int(s["prompt_tokens_max"]), str(s["_id"])))
    chosen = fit[:PER_TIER]
    random.Random(SEED).shuffle(chosen)
    return [dict(sample, index=i) for i, sample in enumerate(chosen)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    data = json.loads(args.data_json.read_text())
    if len(data) != 503:
        raise ValueError(f"官方 LongBench v2 应为 503 题,实际 {len(data)}")
    samples = [to_sample(item) for item in data]
    for sample, tokens in zip(samples, prompt_tokens_max(samples)):
        sample["prompt_tokens_max"] = tokens
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, object] = {
        "version": VERSION,
        "source_sha256": hashlib.sha256(args.data_json.read_bytes()).hexdigest(),
        "seed": SEED,
        "per_tier": PER_TIER,
        "prompt_ratio": PROMPT_RATIO,
        "tokenizers": TOKENIZERS,
        "tiers": {},
    }
    for tier, context in TIERS.items():
        cap = int(context * PROMPT_RATIO)
        chosen = select_tier(samples, cap)
        path = args.out_dir / f"longbench_v2-gate-{tier}-{VERSION}.jsonl"
        body = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in chosen)
        path.write_text(body)
        tokens = [int(s["prompt_tokens_max"]) for s in chosen]
        index["tiers"][tier] = {  # type: ignore[index]
            "file": path.name,
            "context": context,
            "prompt_cap": cap,
            "num_questions": len(chosen),
            "prompt_tokens_min": min(tokens),
            "prompt_tokens_max": max(tokens),
            "prompt_tokens_sum": sum(tokens),
            "sha256": hashlib.sha256(body.encode()).hexdigest(),
            "bytes": len(body.encode()),
            "ids": [s["_id"] for s in chosen],
        }
        print(f"{tier}: cap={cap} n={len(chosen)} tokens[{min(tokens)}..{max(tokens)}] sum={sum(tokens)}")
    (args.out_dir / f"longbench_v2-gate-index-{VERSION}.json").write_text(
        json.dumps(index, indent=1, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
