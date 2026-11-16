"""在本地 CPU 上通过被测服务的 HTTP 接口评测 GSM8K。

参数：endpoint 为服务根地址；data_path 为官方 GSM8K JSONL；result_path 为
评测结果 JSON；config_path 为 task 的 verifier-config.json(提供 gate.kind、
gate.stop_tokens 和 served_model_name)。题数默认 200,可由可选的
gate.num_questions 覆盖(oracle CI 的 smoke 口径用它只跑 10 题);并发 64。

请求协议由 config 的 gate.kind 决定:
- "gsm8k-sglang-generate":SGLang 原生 /generate,5-shot 补全式 prompt,
  stop tokens 来自 gate.stop_tokens;
- "gsm8k-openai-chat":OpenAI 兼容 /v1/chat/completions,0-shot + 固定
  system prompt(要求 '#### <number>' 结尾),temperature 0,不发送 stop
  tokens,与 DSV4 validation harness 的确定性协议一致。仅此协议可显式配置
  gate.max_tokens（默认 1024）和 gate.strict_final_answer（默认 false）；
  strict 模式要求完整最终 content、finish_reason=stop、无工具调用，不读取推理字段。
  gate.reasoning_effort 可选非空字符串，原样传给 OpenAI 请求；省略不改变旧请求。

示例：
    python3 gsm8k_eval.py http://127.0.0.1:8000 \
        gsm8k-official-test.jsonl /tmp/gate-result.json \
        tasks/<task>/tests/verifier-config.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INVALID_ANSWER = -9999999
NUM_SHOTS = 5
# 题数固定 200(gate 口径,标定/评分一致);GSM8K_SMOKE_QUESTIONS 仅供
# 链路连通性 smoke 用——缩到几题就能在几十秒内验证「服务真能推理」,
# 而不必等满量 gate。smoke 的结果不得当基线,accuracy 也无统计意义。
NUM_QUESTIONS = 200
PARALLEL = 64
GSM8K_KINDS = ("gsm8k-sglang-generate", "gsm8k-openai-chat")
CHAT_SYSTEM_PROMPT = (
    "Solve the math problem carefully. End with a separate line in the exact "
    "format '#### <number>'. Do not use tools or external information."
)
CHAT_MAX_TOKENS = 1024


def load_eval_config(config_path: Path) -> dict[str, object]:
    """从 verifier-config.json 读取 GSM8K 评测所需字段;缺失一律 fail loud。"""
    config = json.loads(config_path.read_text())
    gate = config["gate"]
    kind = gate["kind"]
    if kind not in GSM8K_KINDS:
        raise ValueError(f"gate.kind 必须是 {GSM8K_KINDS} 之一：{config_path}")
    stop_tokens = gate["stop_tokens"]
    if not isinstance(stop_tokens, list) or not all(
        isinstance(token, str) for token in stop_tokens
    ):
        raise TypeError(f"gate.stop_tokens 必须是字符串数组：{config_path}")
    num_questions = gate.get("num_questions", NUM_QUESTIONS)
    if (
        not isinstance(num_questions, int)
        or isinstance(num_questions, bool)
        or num_questions < 1
    ):
        raise TypeError(f"gate.num_questions 必须是正整数：{config_path}")
    chat_options = {"max_tokens", "strict_final_answer", "reasoning_effort"}
    if kind != "gsm8k-openai-chat" and chat_options.intersection(gate):
        raise ValueError("gate.max_tokens/strict_final_answer/reasoning_effort 仅支持 gsm8k-openai-chat")
    max_tokens = gate.get("max_tokens", CHAT_MAX_TOKENS)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise TypeError(f"gate.max_tokens 必须为正整数：{config_path}")
    strict_final_answer = gate.get("strict_final_answer", False)
    if not isinstance(strict_final_answer, bool):
        raise TypeError(f"gate.strict_final_answer 必须是布尔值：{config_path}")
    reasoning_effort = gate.get("reasoning_effort")
    if "reasoning_effort" in gate and (
        not isinstance(reasoning_effort, str) or not reasoning_effort.strip()
    ):
        raise TypeError(f"gate.reasoning_effort 必须是非空字符串：{config_path}")
    return {
        "kind": kind,
        "stop_tokens": stop_tokens,
        "served_model_name": str(config["served_model_name"]),
        "num_questions": num_questions,
        "max_tokens": max_tokens,
        "strict_final_answer": strict_final_answer,
        "reasoning_effort": reasoning_effort,
    }


def get_one_example(
    samples: list[dict[str, str]], index: int, include_answer: bool
) -> str:
    """把一条 GSM8K 样本转换为参考 evaluator 使用的文本格式。"""
    prompt = f"Question: {samples[index]['question']}\nAnswer:"
    if include_answer:
        prompt += f" {samples[index]['answer']}"
    return prompt


def get_answer_value(answer: str) -> int:
    """按参考 evaluator 的规则提取文本中最后一个整数答案。"""
    numbers = re.findall(r"\d+", answer.replace(",", ""))
    if not numbers:
        return INVALID_ANSWER
    return ast.literal_eval(numbers[-1])


def get_chat_answer_value(answer: str) -> int:
    """openai_chat 模式的答案提取:优先取最后一个 '####' 标记之后的整数。

    system prompt 要求模型以 '#### <number>' 收尾(GSM8K 官方答案同款格式);
    无标记时回退全文最后一个整数。sglang_generate 5-shot 路径保持原提取规则,
    不受影响(已标定 gate 的口径不变)。
    """
    if "####" in answer:
        tail = get_answer_value(answer.rsplit("####", 1)[1])
        if tail != INVALID_ANSWER:
            return tail
    return get_answer_value(answer)


def _post_json(endpoint_url: str, body: dict[str, object]) -> dict[str, object]:
    """无代理 POST JSON 并解析响应。"""
    request = urllib.request.Request(
        endpoint_url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=900) as response:
        return json.loads(response.read())


def generate_sglang(
    endpoint: str, prompt: str, stop_tokens: list[str]
) -> tuple[int, int]:
    """调用一次 SGLang /generate，返回预测答案和 completion token 数。"""
    result = _post_json(
        f"{endpoint.rstrip('/')}/generate",
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 512,
                "stop": stop_tokens,
            },
        },
    )
    return (
        get_answer_value(result["text"]),
        int(result.get("meta_info", {}).get("completion_tokens", 0)),
    )


def generate_openai_chat(
    endpoint: str,
    question: str,
    served_model_name: str,
    *,
    max_tokens: int = CHAT_MAX_TOKENS,
    strict_final_answer: bool = False,
    reasoning_effort: str | None = None,
    diagnostics: dict[str, str] | None = None,
) -> tuple[int, int]:
    """调用 chat 返回答案和 token 数；strict 校验完整 final，可向 diagnostics 写无效原因。"""
    body = {
        "model": served_model_name,
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    result = _post_json(f"{endpoint.rstrip('/')}/v1/chat/completions", body)
    choice = result["choices"][0]
    message = choice["message"]
    usage = result.get("usage") or {}
    tokens = int(usage.get("completion_tokens", 0))
    if strict_final_answer:
        text = message.get("content")
        reason = None
        if choice.get("finish_reason") == "length":
            reason = "truncated_by_length"
        elif choice.get("finish_reason") != "stop":
            reason = "non_stop_finish"
        elif message.get("tool_calls") or message.get("function_call"):
            reason = "tool_call"
        elif not isinstance(text, str) or not text.strip():
            reason = "empty_final"
        prediction = INVALID_ANSWER if reason else get_chat_answer_value(text)
        if reason is None and prediction == INVALID_ANSWER:
            reason = "unparsed_final"
        if diagnostics is not None:
            diagnostics["final_answer_status"] = reason or "valid_final"
        return prediction, tokens
    text = message.get("content") or message.get("reasoning_content") or ""
    return get_chat_answer_value(text), tokens


def resolve_num_questions(configured: object = NUM_QUESTIONS) -> int:
    """解析 gate 题数；环境变量用于标定 smoke，并优先于 task 配置。"""
    raw = os.environ.get("GSM8K_SMOKE_QUESTIONS", "")
    count = int(raw) if raw else int(configured)
    if count < 1:
        source = "GSM8K_SMOKE_QUESTIONS" if raw else "gate.num_questions"
        raise ValueError(f"{source} 必须为正整数：{count}")
    return min(count, NUM_QUESTIONS)


def evaluate(
    endpoint: str, data_path: Path, eval_config: dict[str, object]
) -> dict[str, object]:
    """评测前 N 题，返回准确率/耗时/吞吐；strict 模式另返回逐原因计数。"""
    samples = [json.loads(line) for line in data_path.read_text().splitlines()]
    num_questions = resolve_num_questions(
        eval_config.get("num_questions", NUM_QUESTIONS)
    )
    if len(samples) < num_questions:
        raise ValueError(f"GSM8K 数据不足 {num_questions} 条：{data_path}")

    kind = eval_config["kind"]
    if kind == "gsm8k-sglang-generate":
        few_shot = "\n\n".join(
            get_one_example(samples, index, True) for index in range(NUM_SHOTS)
        )
        inputs = [
            f"{few_shot}\n\n{get_one_example(samples, index, False)}"
            for index in range(num_questions)
        ]
        stop_tokens = list(eval_config["stop_tokens"])  # type: ignore[arg-type]

        def request(text: str, diagnostics: dict[str, str]) -> tuple[int, int]:
            """发送原 SGLang 请求；此协议不启用 chat 诊断。"""
            return generate_sglang(endpoint, text, stop_tokens)

    else:
        inputs = [samples[index]["question"] for index in range(num_questions)]
        served_model_name = str(eval_config["served_model_name"])

        def request(text: str, diagnostics: dict[str, str]) -> tuple[int, int]:
            """透传 chat 预算及 final 契约，诊断容器为每请求独占。"""
            return generate_openai_chat(
                endpoint, text, served_model_name,
                max_tokens=eval_config.get("max_tokens", CHAT_MAX_TOKENS),
                strict_final_answer=eval_config.get("strict_final_answer", False),
                reasoning_effort=eval_config.get("reasoning_effort"),
                diagnostics=diagnostics,
            )

    labels = [
        get_answer_value(samples[index]["answer"]) for index in range(num_questions)
    ]
    if INVALID_ANSWER in labels:
        raise ValueError("GSM8K 标签中存在无法解析的答案")

    def safe_request(text: str) -> tuple[int, int, str | None]:
        """单请求容错:5xx/断连按无效作答计,不掀翻整轮 gate。

        patched 引擎可能 health 200 但推理即崩(regrade5 实测 agent 补丁
        TypeError 打死 engine core 后全部 500)——此时 gate 应以 accuracy
        崩塌的方式优雅失败,产生 reward=0 的可训练信号,而非 verifier 异常。
        """
        diagnostics: dict[str, str] = {}
        try:
            prediction, tokens = request(text, diagnostics)
            return prediction, tokens, diagnostics.get("final_answer_status")
        except Exception:  # noqa: BLE001 - 网络/服务层失败一律按无效计
            return INVALID_ANSWER, 0, "request_error"

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
        outputs = list(executor.map(safe_request, inputs))
    latency = time.perf_counter() - started_at

    predictions = [prediction for prediction, _, _ in outputs]
    completion_tokens = sum(tokens for _, tokens, _ in outputs)
    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
    invalid = sum(prediction == INVALID_ANSWER for prediction in predictions)
    result: dict[str, object] = {
        "num_questions": num_questions,
        "parallel": PARALLEL,
        "latency_sec": round(latency, 3),
        "accuracy": round(correct / num_questions, 3),
        "invalid_rate": round(invalid / num_questions, 3),
        "output_throughput": round(completion_tokens / latency, 3),
    }
    if eval_config.get("strict_final_answer", False):
        result["final_answer_diagnostics"] = dict(
            Counter(reason for _, _, reason in outputs)
        )
    return result


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
