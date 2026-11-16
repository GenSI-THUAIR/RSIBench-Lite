"""在本地 CPU 上通过被测服务的 HTTP 接口评测 GPQA diamond(gate 口径)。

参数：endpoint 为服务根地址；data_path 为冻结版 GPQA diamond JSONL
(datasets/gate/gpqa_diamond.jsonl,198 题,字段 index/question/options{A-D}/
answer;由官方 simple-evals CSV 以每题独立 RNG——random.Random(index)——
打乱选项后一次性生成,与 the reference GPQA evaluation protocol 同款规则,
入库后不变,保证 gate 可复现);result_path 为评测结果 JSON;config_path 为
task 的 verifier-config.json(提供 gate.kind 和 served_model_name)。

协议(gate.kind = "gpqa-diamond-openai-chat"):OpenAI 兼容
/v1/chat/completions,simple-evals 同款单条 user 消息版式与
"Answer: $LETTER" 抽取(兜底 boxed/裸字母,剥离 thinking 文本),
temperature 0(gate 求方差最小,与模型卡采样口径的差异在 gate.source 里
标注)。输出 schema 与 gsm8k_eval 一致(accuracy/invalid_rate/...),
verifier 的 gate_center/gate_tolerance 逻辑无需变化。

示例：
    python3 gpqa_eval.py http://127.0.0.1:8000 \
        gpqa_diamond.jsonl /tmp/gate-result.json \
        tasks/<task>/tests/verifier-config.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NUM_QUESTIONS = 198
PARALLEL = 64
GPQA_KINDS = ("gpqa-diamond-openai-chat",)
# 生成预算取被测模型官方推荐:LLaDA2.1 模型卡明示 "We recommend using an
# output length of 16384 tokens for most scenarios"(采样推荐 temperature=0,
# top_p/top_k 不设,与本 gate 一致)。预算过小会把长 CoT 截断在 'Answer: X'
# 之前按无效计,系统性压低 accuracy;可用 gate.max_tokens 按 task 覆盖。
DEFAULT_MAX_TOKENS = 16384
REQUEST_ATTEMPTS = 3
REQUEST_TIMEOUT_SEC = 1800
PROMPT_TEMPLATE = """\
Answer the following multiple choice question. The last line of your response \
should be of the following format: 'Answer: $LETTER' (without quotes) where \
LETTER is one of ABCD. Think step by step before answering.

{question}

A) {a}
B) {b}
C) {c}
D) {d}"""

# 依次尝试:标准 Answer: X → boxed{X} → 末尾孤立字母。全按最后一次出现取值。
ANSWER_PATTERNS = [
    re.compile(r"(?i)answer[ \t]*:\s*\$?\(?([ABCD])\)?"),
    re.compile(r"\\boxed\{\(?([ABCD])\)?\}"),
    re.compile(r"(?m)^\s*\(?([ABCD])\)?\s*$"),
]


def load_eval_config(config_path: Path) -> dict[str, object]:
    """从 verifier-config.json 读取 GPQA 评测所需字段;缺失一律 fail loud。"""
    config = json.loads(config_path.read_text())
    kind = config["gate"]["kind"]
    if kind not in GPQA_KINDS:
        raise ValueError(f"gate.kind 必须是 {GPQA_KINDS} 之一：{config_path}")
    num_questions = config["gate"].get("num_questions", NUM_QUESTIONS)
    if (
        not isinstance(num_questions, int)
        or isinstance(num_questions, bool)
        or not 1 <= num_questions <= NUM_QUESTIONS
    ):
        raise TypeError(
            f"gate.num_questions 必须是 1..{NUM_QUESTIONS} 的整数：{config_path}"
        )
    return {
        "kind": kind,
        "served_model_name": str(config["served_model_name"]),
        "max_tokens": int(config["gate"].get("max_tokens", DEFAULT_MAX_TOKENS)),
        "num_questions": num_questions,
    }


def format_prompt(sample: dict[str, object]) -> str:
    """按 simple-evals 版式渲染题干与选项(单条 user 消息)。"""
    options = cast_options(sample)
    return PROMPT_TEMPLATE.format(
        question=sample["question"],
        a=options["A"],
        b=options["B"],
        c=options["C"],
        d=options["D"],
    )


def cast_options(sample: dict[str, object]) -> dict[str, str]:
    options = sample["options"]
    if not isinstance(options, dict) or sorted(options) != ["A", "B", "C", "D"]:
        raise ValueError(f"样本 options 必须含 A-D 四项：{sample.get('index')}")
    return {key: str(value) for key, value in options.items()}


def cast_int(value: object) -> int:
    if not isinstance(value, (int, float)) or int(value) <= 0:
        raise ValueError(f"max_tokens 必须为正整数：{value!r}")
    return int(value)


def extract_answer(text: str) -> str | None:
    """按 simple-evals 规则抽取最终答案字母;thinking 文本先剥离。"""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    return None


def _post_json(endpoint_url: str, body: dict[str, object]) -> dict[str, object]:
    """无代理 POST JSON 并解析响应。"""
    request = urllib.request.Request(
        endpoint_url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=REQUEST_TIMEOUT_SEC) as response:
        return json.loads(response.read())


def generate_openai_chat(
    endpoint: str, prompt: str, served_model_name: str, max_tokens: int
) -> tuple[str | None, int]:
    """调用一次 /v1/chat/completions，返回预测选项和 completion token 数。

    网络层错误重试(与 cookbooks 版同款);重试耗尽按无效作答计,不掀翻
    整轮 gate。
    """
    # 官方采样推荐 top_p/top_k=None,故不发这两个字段。
    body = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    last_error: Exception | None = None
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            result = _post_json(f"{endpoint.rstrip('/')}/v1/chat/completions", body)
            message = result["choices"][0]["message"]
            text = message.get("content") or ""
            reasoning = (
                message.get("reasoning_content") or message.get("reasoning") or ""
            )
            answer = extract_answer(text) or extract_answer(reasoning)
            usage = result.get("usage") or {}
            return answer, int(usage.get("completion_tokens", 0))
        except Exception as error:  # noqa: BLE001 - 网络层一律重试
            last_error = error
            time.sleep(5 * (attempt + 1))
    del last_error
    return None, 0


def evaluate(
    endpoint: str, data_path: Path, eval_config: dict[str, object]
) -> dict[str, float | int]:
    """并发评测前 N 题(默认全部 198 题)，返回准确率、耗时和输出吞吐。"""
    samples = [json.loads(line) for line in data_path.read_text().splitlines()]
    if len(samples) != NUM_QUESTIONS:
        raise ValueError(f"GPQA diamond 数据必须正好 {NUM_QUESTIONS} 条：{data_path}")
    num_questions = int(eval_config.get("num_questions", NUM_QUESTIONS))
    if not 1 <= num_questions <= NUM_QUESTIONS:
        raise ValueError(f"num_questions 必须在 1..{NUM_QUESTIONS}：{num_questions}")
    samples = samples[:num_questions]
    served_model_name = str(eval_config["served_model_name"])
    max_tokens = int(cast_int(eval_config.get("max_tokens", DEFAULT_MAX_TOKENS)))
    inputs = [format_prompt(sample) for sample in samples]
    labels = [str(sample["answer"]) for sample in samples]

    def request(prompt: str) -> tuple[str | None, int]:
        return generate_openai_chat(endpoint, prompt, served_model_name, max_tokens)

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
        outputs = list(executor.map(request, inputs))
    latency = time.perf_counter() - started_at

    predictions = [prediction for prediction, _ in outputs]
    completion_tokens = sum(tokens for _, tokens in outputs)
    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
    invalid = sum(prediction is None for prediction in predictions)
    return {
        "num_questions": num_questions,
        "parallel": PARALLEL,
        "latency_sec": round(latency, 3),
        "accuracy": round(correct / num_questions, 3),
        "invalid_rate": round(invalid / num_questions, 3),
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

    result = evaluate(args.endpoint, args.data_path, load_eval_config(args.config_path))
    args.result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
