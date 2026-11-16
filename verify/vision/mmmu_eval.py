"""在本地 CPU 上通过 OpenAI 兼容接口评测 MMMU(MCQ,带图)。

参数：endpoint 为服务根地址；data_path 为 prepare_mmmu.py 导出的自包含
JSONL(图片 base64 内嵌);result_path 为评测结果 JSON;config_path 为 task
的 verifier-config.json(提供 served_model_name)。

协议:/v1/chat/completions,每题图片按序作为 image_url(data URI)内容块,
temperature 0 确定性采样,'Answer: $LETTER' 抽取(兜底 boxed/裸字母,与
gpqa_diamond_eval 同规则)。输出 schema 与 gsm8k_eval 一致,直接进
clean_benchmark 的 gate 汇总。

示例：
    python3 mmmu_eval.py http://127.0.0.1:8000 \
        /mnt/rsibench/rsibench/mmmu_val_mcq.jsonl \
        /tmp/gate-result.json tasks/<task>/tests/verifier-config.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PARALLEL = 16
MAX_TOKENS = 8192
LETTERS = "ABCDEFGHIJ"
# 依次尝试:标准 Answer: X → boxed{X} → 末尾孤立字母。全按最后一次出现取值。
ANSWER_PATTERNS = [
    re.compile(r"(?i)answer[ \t]*:?\s*\$?\(?([A-J])\)?(?![a-z])"),
    re.compile(r"\\boxed\{\(?([A-J])\)?\}"),
    re.compile(r"(?m)^\s*\(?([A-J])\)?\s*$"),
]
PROMPT_SUFFIX = (
    "\n\nAnswer the multiple choice question. The last line of your response "
    "must be of the following format: 'Answer: $LETTER' (without quotes) "
    "where $LETTER is the letter of the correct option."
)


def extract_answer(text: str) -> str | None:
    """按 gpqa_diamond_eval 同款规则抽取答案字母。"""
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    return None


def build_messages(sample: dict[str, object]) -> list[dict[str, object]]:
    """一条 MMMU MCQ 组装为带图 chat 消息。"""
    lines = [str(sample["question"]), ""]
    for index, option in enumerate(sample["options"]):  # type: ignore[arg-type]
        lines.append(f"{LETTERS[index]}) {option}")
    content: list[dict[str, object]] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{image['mime']};base64,{image['b64']}"},
        }
        for image in sample["images"]  # type: ignore[union-attr]
    ]
    content.append({"type": "text", "text": "\n".join(lines) + PROMPT_SUFFIX})
    return [{"role": "user", "content": content}]


def ask(endpoint: str, served_model_name: str, sample: dict[str, object]) -> tuple[bool, bool, bool, int]:
    """单题请求,返回 (是否答对, 是否无法解析, 是否被截断, completion token 数)。"""
    body = {
        "model": served_model_name,
        "messages": build_messages(sample),
        "temperature": 0,
        "top_p": 1,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=900) as response:
        result = json.loads(response.read())
    choice = result["choices"][0]
    message = choice["message"]
    text = message.get("content") or message.get("reasoning_content") or ""
    predicted = extract_answer(text)
    usage = result.get("usage") or {}
    return (
        predicted == sample["answer"],
        predicted is None,
        choice.get("finish_reason") == "length",
        int(usage.get("completion_tokens", 0)),
    )


def evaluate(
    endpoint: str, data_path: Path, eval_config: dict[str, object]
) -> dict[str, float | int]:
    """并发评测 MCQ(默认全量;可选 num_questions 只取前 N 条,smoke 口径用),
    返回与 gsm8k_eval 同 schema 的汇总。"""
    samples = [json.loads(line) for line in data_path.read_text().splitlines()]
    num_questions = eval_config.get("num_questions")
    if num_questions is not None:
        samples = samples[: int(num_questions)]
    if not samples:
        raise ValueError(f"MMMU 数据为空：{data_path}")
    served_model_name = str(eval_config["served_model_name"])

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
        outputs = list(
            executor.map(
                lambda sample: ask(endpoint, served_model_name, sample), samples
            )
        )
    latency = time.perf_counter() - started_at

    correct = sum(is_correct for is_correct, _, _, _ in outputs)
    invalid = sum(unparsed for _, unparsed, _, _ in outputs)
    truncated = sum(is_truncated for _, _, is_truncated, _ in outputs)
    completion_tokens = sum(tokens for _, _, _, tokens in outputs)
    return {
        "num_questions": len(samples),
        "parallel": PARALLEL,
        "latency_sec": round(latency, 3),
        "accuracy": round(correct / len(samples), 4),
        "invalid_rate": round(invalid / len(samples), 4),
        "truncated_rate": round(truncated / len(samples), 4),
        "output_throughput": round(completion_tokens / latency, 3),
    }


def main() -> None:
    """解析四个位置参数，执行评测并写入结果文件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint")
    parser.add_argument("data_path", type=Path)
    parser.add_argument("result_path", type=Path)
    parser.add_argument("config_path", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config_path.read_text())
    result = evaluate(
        args.endpoint,
        args.data_path,
        {
            "served_model_name": str(config["served_model_name"]),
            "num_questions": config["gate"].get("num_questions"),
        },
    )
    args.result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
