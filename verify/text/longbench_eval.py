"""在本地 CPU 上通过被测服务的 HTTP 接口评测 LongBench v2(gate 口径)。

参数：endpoint 为服务根地址；data_path 为冻结版 LongBench v2 子集 JSONL
(由 scripts/prepare/build_longbench_gate_data.py 按长度档从官方 THUDM/LongBench-v2
data.json 抽样生成,字段 index/_id/domain/sub_domain/difficulty/length/
question/choices{A-D}/answer/context,入库后不变,保证 gate 可复现);
result_path 为评测结果 JSON;config_path 为 task 的 verifier-config.json
(提供 gate.kind、served_model_name,可选 gate.max_tokens / gate.num_questions /
gate.max_concurrency / gate.reasoning_effort)。

协议(gate.kind = "longbench-v2-openai-chat"):OpenAI 兼容
/v1/chat/completions,官方 prompts/0shot.txt 单条 user 消息版式(长文本置于
<text> 标签内,四选一,要求以 "The correct answer is (X)" 作答),官方
extract_answer 正则抽取(兜底 boxed / 裸字母,剥离 thinking 文本),
temperature 0(官方 pred.py 用 0.1,gate 求方差最小)。并发默认 4——单条
prompt 数万到二十余万 token,高并发会把 KV cache 打满。输出 schema 与
gsm8k_eval / gpqa_eval 一致(accuracy / invalid_rate / ...),verifier 的
gate_center / gate_tolerance / comparison 逻辑无需变化。

示例：
    python3 longbench_eval.py http://127.0.0.1:8000 \
        longbench_v2-gate-128k-v1.jsonl /tmp/gate-result.json \
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

LONGBENCH_KINDS = ("longbench-v2-openai-chat",)
DEFAULT_PARALLEL = 4
# 官方非 CoT 口径 max_new_tokens=128;本 gate 面向的多数被测模型默认带
# thinking,答案在 reasoning 之后,预算过小会把 final 截断在答案之前按无效计。
# 可用 gate.max_tokens 按 task 覆盖。
DEFAULT_MAX_TOKENS = 4096
REQUEST_ATTEMPTS = 3
REQUEST_TIMEOUT_SEC = 3600
# 无效作答诊断只留尾部片段,gate-result.json 保持小体积。
INVALID_SAMPLE_CHARS = 200
INVALID_SAMPLES_KEPT = 5
CHOICES = ("A", "B", "C", "D")
# 官方 prompts/0shot.txt 原文;$DOC$/$Q$/$C_X$ 由 format_prompt 替换。
PROMPT_TEMPLATE = """\
Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""

# 前两条是官方 pred.py extract_answer 的正则;后两条兜底 boxed / 末尾孤立字母。
# 全按最后一次出现取值(模型复述题干或自我修正时以最终结论为准)。
ANSWER_PATTERNS = [
    re.compile(r"The correct answer is \(([A-D])\)"),
    re.compile(r"The correct answer is ([A-D])"),
    re.compile(r"\\boxed\{\(?([A-D])\)?\}"),
    re.compile(r"(?m)^\s*\(?([A-D])\)?\s*$"),
]


def load_eval_config(config_path: Path) -> dict[str, object]:
    """从 verifier-config.json 读取 LongBench v2 评测所需字段;缺失一律 fail loud。"""
    config = json.loads(config_path.read_text())
    gate = config["gate"]
    kind = gate["kind"]
    if kind not in LONGBENCH_KINDS:
        raise ValueError(f"gate.kind 必须是 {LONGBENCH_KINDS} 之一：{config_path}")
    num_questions = gate.get("num_questions")
    if num_questions is not None and (
        not isinstance(num_questions, int)
        or isinstance(num_questions, bool)
        or num_questions < 1
    ):
        raise TypeError(f"gate.num_questions 必须是正整数：{config_path}")
    parallel = gate.get("max_concurrency", DEFAULT_PARALLEL)
    if not isinstance(parallel, int) or isinstance(parallel, bool) or parallel < 1:
        raise TypeError(f"gate.max_concurrency 必须是正整数：{config_path}")
    reasoning_effort = gate.get("reasoning_effort")
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str) or not reasoning_effort
    ):
        raise TypeError(f"gate.reasoning_effort 必须是非空字符串：{config_path}")
    return {
        "kind": kind,
        "served_model_name": str(config["served_model_name"]),
        "max_tokens": cast_int(gate.get("max_tokens", DEFAULT_MAX_TOKENS)),
        "num_questions": num_questions,
        "parallel": parallel,
        "reasoning_effort": reasoning_effort,
    }


def cast_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"max_tokens 必须为正整数：{value!r}")
    return value


def cast_choices(sample: dict[str, object]) -> dict[str, str]:
    choices = sample["choices"]
    if not isinstance(choices, dict) or sorted(choices) != list(CHOICES):
        raise ValueError(f"样本 choices 必须含 A-D 四项：{sample.get('_id')}")
    return {key: str(value) for key, value in choices.items()}


def format_prompt(sample: dict[str, object]) -> str:
    """按官方 0shot 模板渲染:上下文、题干与四个选项,各字段先 strip。"""
    choices = cast_choices(sample)
    return (
        PROMPT_TEMPLATE.replace("$DOC$", str(sample["context"]).strip())
        .replace("$Q$", str(sample["question"]).strip())
        .replace("$C_A$", choices["A"].strip())
        .replace("$C_B$", choices["B"].strip())
        .replace("$C_C$", choices["C"].strip())
        .replace("$C_D$", choices["D"].strip())
    )


def extract_answer(text: str) -> str | None:
    """按官方规则抽取答案字母(先去掉 markdown 星号);thinking 文本先剥离。"""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.replace("*", "")
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
    endpoint: str,
    prompt: str,
    served_model_name: str,
    max_tokens: int,
    reasoning_effort: str | None = None,
) -> tuple[str | None, int, bool, str | None]:
    """调用一次 /v1/chat/completions，返回(预测选项, completion token 数, 是否被 max_tokens 截断, 无效原因)。

    网络层错误重试;重试耗尽按无效作答计,不掀翻整轮 gate。答案优先从
    content 抽取,content 里没有时再看 reasoning_content(被截断的思考)。
    无效原因用于事后诊断:"request_error: <最后一次异常>" 或
    "no_answer(finish=<finish_reason>): <content 头部>";有效作答为 None。
    """
    body: dict[str, object] = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    last_error = "unknown"
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            result = _post_json(f"{endpoint.rstrip('/')}/v1/chat/completions", body)
            choice = result["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
            reasoning = (
                message.get("reasoning_content") or message.get("reasoning") or ""
            )
            answer = extract_answer(text) or extract_answer(reasoning)
            usage = result.get("usage") or {}
            finish = choice.get("finish_reason")
            reason = None
            if answer is None:
                head = (text or reasoning)[-INVALID_SAMPLE_CHARS:].replace("\n", " ")
                reason = f"no_answer(finish={finish}): {head}"
            return answer, int(usage.get("completion_tokens", 0)), finish == "length", reason
        except Exception as error:  # noqa: BLE001 - 网络层一律重试
            last_error = f"{type(error).__name__}: {str(error)[:INVALID_SAMPLE_CHARS]}"
            time.sleep(5 * (attempt + 1))
    return None, 0, False, f"request_error: {last_error}"


def evaluate(
    endpoint: str, data_path: Path, eval_config: dict[str, object]
) -> dict[str, object]:
    """并发评测前 N 题(默认文件全部)，返回准确率、无效率、截断率、耗时和输出吞吐。"""
    # 只按 \n 切分:长文本里含 U+2028/U+2029 等 Unicode 行分隔符,str.splitlines() 会
    # 把它们也当换行,把一条记录截成两半(2026-09-16 muse-glimmer 标定实测踩坑)。
    samples = [json.loads(line) for line in data_path.read_text().split("\n") if line]
    if not samples:
        raise ValueError(f"LongBench v2 数据为空：{data_path}")
    num_questions = eval_config.get("num_questions")
    if num_questions is None:
        num_questions = len(samples)
    num_questions = int(num_questions)
    if not 1 <= num_questions <= len(samples):
        raise ValueError(f"num_questions 必须在 1..{len(samples)}：{num_questions}")
    samples = samples[:num_questions]
    served_model_name = str(eval_config["served_model_name"])
    max_tokens = cast_int(eval_config.get("max_tokens", DEFAULT_MAX_TOKENS))
    parallel = int(eval_config.get("parallel", DEFAULT_PARALLEL))
    reasoning_effort = eval_config.get("reasoning_effort")
    inputs = [format_prompt(sample) for sample in samples]
    labels = [str(sample["answer"]) for sample in samples]
    for label in labels:
        if label not in CHOICES:
            raise ValueError(f"样本 answer 必须是 A-D：{label!r}")

    def request(prompt: str) -> tuple[str | None, int, bool, str | None]:
        return generate_openai_chat(
            endpoint,
            prompt,
            served_model_name,
            max_tokens,
            None if reasoning_effort is None else str(reasoning_effort),
        )

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        outputs = list(executor.map(request, inputs))
    latency = time.perf_counter() - started_at

    predictions = [prediction for prediction, _, _, _ in outputs]
    completion_tokens = sum(tokens for _, tokens, _, _ in outputs)
    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
    invalid = sum(prediction is None for prediction in predictions)
    truncated = sum(flag for _, _, flag, _ in outputs)
    reasons = [reason for _, _, _, reason in outputs if reason is not None]
    invalid_reasons: dict[str, int] = {}
    for reason in reasons:
        key = reason.split(":", 1)[0].split("(", 1)[0]
        invalid_reasons[key] = invalid_reasons.get(key, 0) + 1
    by_length: dict[str, list[int]] = {}
    for sample, prediction, label in zip(samples, predictions, labels):
        bucket = by_length.setdefault(str(sample.get("length", "unknown")), [0, 0])
        bucket[0] += int(prediction == label)
        bucket[1] += 1
    return {
        "num_questions": num_questions,
        "parallel": parallel,
        "max_tokens": max_tokens,
        "latency_sec": round(latency, 3),
        "accuracy": round(correct / num_questions, 3),
        "invalid_rate": round(invalid / num_questions, 3),
        "truncated_rate": round(truncated / num_questions, 3),
        "output_throughput": round(completion_tokens / latency, 3),
        "accuracy_by_length": {
            key: round(hit / total, 3) for key, (hit, total) in sorted(by_length.items())
        },
        # 无效作答按类别计数(request_error / no_answer)并保留前几条样本,便于区分
        # 服务崩溃、格式不符与思考截断(2026-09-17 glm5p3flash 标定时 serve 崩溃、94% 无效却无从判断)。
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "invalid_samples": reasons[:INVALID_SAMPLES_KEPT],
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
