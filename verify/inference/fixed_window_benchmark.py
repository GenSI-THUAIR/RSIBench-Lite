#!/usr/bin/env python3
"""C1 固定时间窗回放；可在计时前预热，只计窗口内连续累计 usage，不执行正确性 gate。

参数：endpoint output_dir verifier-config.json（完整 scoring JSON）。
示例：python fixed_window_benchmark.py http://localhost:8000 /tmp/window verifier-config.json
窗口与引擎总数必须显式配置；词表默认从 model_path 本地加载，可用 bench_serving.vocab_json
指定实际 get_vocab()/all_special_ids 导出文件。warmup_requests 默认0；启用后重复首条输入，输出最多32token，
warmup_timeout_sec 默认600秒（全部预热的总预算）。预热须完整成功且排空，时间和token不计分。
调用方负责独占服务、先窗口后正确性，并在正确性前调用 wait_for_idle 排空请求。
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import base64
from collections import deque
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import re
import time
import traceback
import typing
from urllib.parse import urlsplit, urlunsplit

import aiohttp


class ProtocolError(ValueError):
    """服务/计数不满足协议；保留证据但不产生可评分吞吐。"""


class WorkloadExhausted(ProtocolError):
    """无节流固定窗口的工作集提前耗尽；不能空等后冒充有效吞吐。"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _base_url(endpoint):
    if not isinstance(endpoint, str):
        raise ValueError("endpoint must be an HTTP(S) URL")
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.query or parts.fragment:
        raise ValueError("endpoint must be an HTTP(S) URL without query or fragment")
    path = parts.path.rstrip("/")
    for suffix in ("/v1/completions", "/v1"):
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclasses.dataclass(frozen=True)
class WindowConfig:
    model_path: str
    served_model_name: str
    duration_sec: float
    dataset_path: Path
    context_length: int
    slowdown_factor: float
    seed: int
    num_prompts: int | None
    expected_engines: int
    trace_sha256: str | None
    vocab_json: Path | None
    warmup_requests: int
    warmup_timeout_sec: float
    backend: str = "vllm"
    full_duration: bool = False
    idle_backend: str = "vllm"
    arrival_mode: str = "trace"


def _parse_config(config):
    if not isinstance(config, dict) or not isinstance(config.get("bench_serving"), dict):
        raise ValueError("expected full scoring JSON with bench_serving")
    bench = config["bench_serving"]
    arrival_mode = bench.get("arrival_mode", "trace")
    if not isinstance(arrival_mode, str) or arrival_mode not in {"trace", "asap"}:
        raise ValueError("bench_serving.arrival_mode must be trace or asap")
    duration = _positive_float(bench.get("window_duration_sec"), "window_duration_sec")
    if duration > 3600:
        raise ValueError("window_duration_sec must be at most 3600")
    if bench.get("backend") not in {"vllm", "sglang"} or bench.get("dataset_name") != "mooncake":
        raise ValueError("fixed windows require backend=vllm/sglang and dataset_name=mooncake")
    if type(bench.get("window_full_duration", False)) is not bool:
        raise ValueError("window_full_duration must be a boolean")
    if type(bench.get("max_concurrency")) is not int or bench["max_concurrency"] != 1:
        raise ValueError("fixed windows require max_concurrency=1")
    for name in ("model_path", "served_model_name"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    if type(bench.get("seed")) is not int:
        raise ValueError("bench_serving.seed must be an integer")
    cap = bench.get("num_prompts")
    if "num_prompts" in bench:
        _positive_int(cap, "num_prompts")
    engines = _positive_int(bench.get("window_expected_engines"), "window_expected_engines")
    idle_backend = bench.get("window_idle_backend", bench["backend"])
    if idle_backend != bench["backend"] and not (idle_backend == "trtllm" and bench["backend"] == "vllm"):
        raise ValueError("window_idle_backend must match backend or use trtllm with vllm wire protocol")
    if idle_backend == "trtllm" and engines != 1:
        raise ValueError("TRT-LLM idle metrics currently require one serving engine")
    warmup_requests = bench.get("warmup_requests", 0)
    if type(warmup_requests) is not int or warmup_requests < 0:
        raise ValueError("warmup_requests must be a nonnegative integer")
    warmup_timeout = _positive_float(bench.get("warmup_timeout_sec", 600), "warmup_timeout_sec")
    trace_hash = bench.get("trace_sha256")
    if trace_hash is not None and (not isinstance(trace_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", trace_hash)):
        raise ValueError("trace_sha256 must be a SHA256 hex digest")
    vocab_path = bench.get("vocab_json")
    if vocab_path is not None and (not isinstance(vocab_path, str) or not vocab_path):
        raise ValueError("vocab_json must be a nonempty export path")

    # 完整 scoring JSON 中的 script/timeout/random 元数据不改变 mooncake 请求。
    # 请求参数只能来自以下四个 flag，拒绝覆盖 stream/seed/budget 等协议参数。
    extra = bench.get("extra_args", [])
    if not isinstance(extra, list) or any(not isinstance(arg, str) for arg in extra):
        raise ValueError("bench_serving.extra_args must be a list of strings")
    allowed = {"--dataset-path", "--mooncake-context-length", "--mooncake-slowdown-factor", "--mooncake-num-rounds"}
    flags = {}
    index = 0
    while index < len(extra):
        flag, sep, value = extra[index].partition("=")
        if flag not in allowed or flag in flags:
            raise ValueError(f"unsupported or duplicate window extra_arg: {flag}")
        if not sep:
            index += 1
            if index == len(extra) or extra[index].startswith("--"):
                raise ValueError(f"missing value for {flag}")
            value = extra[index]
        if not value:
            raise ValueError(f"missing value for {flag}")
        flags[flag] = value
        index += 1
    if "--dataset-path" not in flags or "--mooncake-context-length" not in flags:
        raise ValueError("window replay requires --dataset-path and --mooncake-context-length")
    context = _positive_int(int(flags["--mooncake-context-length"]), "mooncake-context-length")
    slowdown = _positive_float(float(flags.get("--mooncake-slowdown-factor", "1.0")), "mooncake-slowdown-factor")
    if int(flags.get("--mooncake-num-rounds", "1")) != 1:
        raise ValueError("fixed windows require --mooncake-num-rounds=1")
    return WindowConfig(config["model_path"], config["served_model_name"], duration,
                        Path(flags["--dataset-path"]), context, slowdown, bench["seed"], cap,
                        engines, trace_hash.lower() if trace_hash else None,
                        Path(vocab_path) if vocab_path else None, warmup_requests, warmup_timeout,
                        bench["backend"], bench.get("window_full_duration", False), idle_backend, arrival_mode)


def load_workload(config: WindowConfig):
    """提取 sibling 原编译器，全量预检/编译后才截取前 N 行；不加载模型权重。"""
    raw_trace = config.dataset_path.read_bytes()
    trace_hash = digest(raw_trace)
    if config.trace_sha256 is not None and trace_hash != config.trace_sha256:
        raise ValueError(f"original trace SHA256 mismatch: {config.dataset_path}")
    records = [json.loads(line) for line in raw_trace.splitlines() if line.strip()]
    if config.vocab_json is not None:
        raw_vocab = config.vocab_json.read_bytes()
        exported = json.loads(raw_vocab)
        vocab, special = exported["vocab"], exported["all_special_ids"]
        vocab_source = {"path": str(config.vocab_json.resolve()), "file_sha256": digest(raw_vocab)}
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True, trust_remote_code=True)
        vocab, special = tokenizer.get_vocab(), tokenizer.all_special_ids
        vocab_source = {"path": str(Path(config.model_path).resolve()), "loader": "AutoTokenizer(local_files_only=True)"}
    if not isinstance(vocab, dict) or not vocab or any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in vocab.items()):
        raise ValueError("invalid exported tokenizer vocabulary")
    if not isinstance(special, list) or any(type(i) is not int or i < 0 for i in special):
        raise ValueError("invalid exported all_special_ids")

    class Vocabulary:
        all_special_ids = special

        def get_vocab(self):
            return vocab

    bench_path = Path(__file__).with_name("vendored_bench_serving.py")
    raw_source = bench_path.read_bytes()
    source = raw_source.decode("utf-8")
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {"DatasetRow", "compile_mooncake_requests"}:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"MOONCAKE_PROTOCOL", "MOONCAKE_BLOCK_SIZE", "MOONCAKE_ID_BACKENDS"} for t in node.targets):
            selected.append(node)
    if len(selected) != 5:
        raise ValueError("vendored compiler definitions changed; refusing to invent replacement workload")
    namespace = dict(vars(typing), dataclass=dataclasses.dataclass, PreTrainedTokenizerBase=object,
                     hashlib=hashlib, json=json, math=math, __name__=__name__)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(bench_path), "exec"), namespace)
    rows = namespace["compile_mooncake_requests"](records, Vocabulary(), backend=config.backend,
                                                context_length=config.context_length, seed=config.seed, num_rounds=1)
    rows = rows[:config.num_prompts]
    metadata = {
        "trace_path": str(config.dataset_path), "trace_sha256": trace_hash,
        "trace_request_count": len(records), "trace_total_requests": len(records), "selected_request_count": len(rows),
        "compiler_path": str(bench_path), "compiler_file_sha256": digest(raw_source),
        "compiler_definitions_sha256": digest("\n".join(ast.get_source_segment(source, node) for node in selected).encode()),
        "tokenizer": {**vocab_source, "vocab_and_special_ids_sha256": digest(canonical({"vocab": vocab, "all_special_ids": special})),
                      "vocab_id_count": len(set(vocab.values())), "all_special_ids": special},
    }
    return rows, metadata, raw_trace


_METRIC_NAMES = ("vllm:num_requests_running", "vllm:num_requests_waiting")
_IDLE_METRICS = {
    "vllm": ("/metrics", _METRIC_NAMES),
    "sglang": ("/metrics", ("sglang:num_running_reqs", "sglang:num_queue_reqs")),
    "trtllm": ("/prometheus/metrics", ("trtllm_num_requests_running", "trtllm_num_requests_waiting")),
}
_METRIC_LINE = re.compile(r'(vllm:num_requests_(?:running|waiting))(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?\s*')
_METRIC_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*("(?:[^"\\]|\\["\\n])*")\s*(?:,\s*|$)')
_IDLE_POLL_SEC = 0.25


def _idle_gauges(text, expected_engines, backend="vllm"):
    """必须看到同一组 engine 的 running/waiting；不能把缺失 gauge 当成零。"""
    if backend not in _IDLE_METRICS:
        raise ValueError(f"unsupported idle backend: {backend}")
    if backend != "vllm":
        # wire协议不决定引擎指标；TRT-LLM的/metrics是iteration JSON，不能当Prometheus。
        names = _IDLE_METRICS[backend][1]
        text = "\n".join(line for line in text.splitlines() if any(
            re.match(re.escape(name) + r"(?:\{|\s|$)", line.strip()) for name in names))
        text = text.replace(names[0], _METRIC_NAMES[0]).replace(names[1], _METRIC_NAMES[1])
    gauges = {name: {} for name in _METRIC_NAMES}
    for line in text.splitlines():
        line = line.strip()
        if not any(re.match(re.escape(name) + r"(?:\{|\s|$)", line) for name in _METRIC_NAMES):
            continue
        match = _METRIC_LINE.fullmatch(line)
        if match is None:
            raise ProtocolError(f"malformed idle metric: {line}")
        name, raw_labels, value = match.groups()
        labels, offset = {}, 0
        raw_labels = raw_labels or ""
        while offset < len(raw_labels):
            label = _METRIC_LABEL.match(raw_labels, offset)
            if label is None or label[1] in labels:
                raise ProtocolError(f"malformed/duplicate idle metric label: {line}")
            labels[label[1]] = json.loads(label[2])
            offset = label.end()
        if "engine" in labels and "engine_id" in labels and labels["engine"] != labels["engine_id"]:
            raise ProtocolError("conflicting engine labels in idle metric")
        engine = labels.get("engine", labels.get("engine_id"))
        if engine is None:
            if expected_engines != 1:
                raise ProtocolError("idle metrics must identify every expected engine")
            engine = "unlabeled"
        if not engine or engine in gauges[name]:
            raise ProtocolError(f"empty/duplicate engine in {name}")
        try:
            number = float(value)
        except ValueError as exc:
            raise ProtocolError(f"invalid idle metric value: {line}") from exc
        if not math.isfinite(number) or number < 0:
            raise ProtocolError(f"idle metrics must be finite and nonnegative: {line}")
        gauges[name][engine] = number
    if any(len(values) != expected_engines for values in gauges.values()):
        raise ProtocolError(f"idle metrics require {expected_engines} engines in both gauge families")
    if set(gauges[_METRIC_NAMES[0]]) != set(gauges[_METRIC_NAMES[1]]):
        raise ProtocolError("running/waiting idle metric engine sets differ")
    return gauges


async def _wait_for_idle(endpoint, evidence_path, *, expected_engines, timeout_sec=180, backend="vllm"):
    evidence_path = Path(evidence_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    if evidence_path.exists() or evidence_path.is_symlink():
        raise FileExistsError(f"idle evidence already exists: {evidence_path}")
    loop = asyncio.get_running_loop()
    started = loop.time()
    evidence = {"status": "checking", "started_unix_sec": time.time(), "samples": []}
    try:
        expected_engines = _positive_int(expected_engines, "expected_engines")
        timeout_sec = _positive_float(timeout_sec, "timeout_sec")
        if backend not in _IDLE_METRICS:
            raise ValueError(f"unsupported idle backend: {backend}")
        base = _base_url(endpoint)
        evidence.update(endpoint=base, expected_engines=expected_engines, timeout_sec=timeout_sec)
        deadline = started + timeout_sec

        async def get(session, path):
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("idle check timed out")
            # aiohttp normally rounds long timeouts up to whole seconds; not here.
            timeout = aiohttp.ClientTimeout(total=min(10, remaining), ceil_threshold=math.inf)
            async with session.get(base + path, timeout=timeout, allow_redirects=False) as response:
                if response.status != 200:
                    raise ProtocolError(f"idle check {path}: HTTP {response.status}")
                return await response.text()

        async with asyncio.timeout_at(deadline):
            async with aiohttp.ClientSession(trust_env=False) as session:
                await get(session, "/health")
                evidence["health_status"] = 200
                models = json.loads(await get(session, "/v1/models"))
                if not isinstance(models, dict) or not isinstance(models.get("data"), list) or not models["data"] or any(
                    not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"] for item in models["data"]
                ):
                    raise ProtocolError("idle check /v1/models has no valid model IDs")
                evidence["models_response"] = models
                while True:
                    sample = {"offset_sec": loop.time() - started, "unix_sec": time.time()}
                    evidence["samples"].append(sample)
                    metric_path, names = _IDLE_METRICS[backend]
                    sample["metrics_path"] = metric_path
                    text = await get(session, metric_path)
                    sample["metric_lines"] = [line for line in text.splitlines() if any(line.startswith(name) for name in names)]
                    sample["gauges"] = gauges = _idle_gauges(text, expected_engines, backend)
                    sample["idle"] = all(value == 0 for family in gauges.values() for value in family.values())
                    if sample["idle"]:
                        evidence["status"] = "idle"
                        return evidence
                    await asyncio.sleep(_IDLE_POLL_SEC)
    except asyncio.CancelledError:
        evidence["status"] = "cancelled"
        raise
    except Exception as exc:
        evidence.update(status="timeout" if isinstance(exc, TimeoutError) else "failure",
                        error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        evidence["actual_elapsed_wall_sec"] = loop.time() - started
        evidence["finalized_unix_sec"] = time.time()
        write_json(evidence_path, evidence)


def wait_for_idle(endpoint: str, evidence_path: Path, *, expected_engines: int, timeout_sec: float = 180, backend: str = "vllm") -> dict:
    """同步排空检查；仅当所有引擎两类 gauge 均为零才返回，否则落盘后抛错。"""
    return asyncio.run(_wait_for_idle(endpoint, evidence_path, expected_engines=expected_engines, timeout_sec=timeout_sec, backend=backend))


class SSEDecoder:
    """保留跨网络块的 SSE 边界和 UTF-8 字节；只交付完整事件。"""

    def __init__(self):
        self.buffer = b""

    def feed(self, chunk):
        self.buffer += chunk
        events = []
        while (match := re.search(rb"\r?\n\r?\n", self.buffer)) is not None:
            block = self.buffer[:match.start()]
            self.buffer = self.buffer[match.end():]
            # SSE只认CR/LF；str.splitlines会误拆JSON字符串中的U+0085/U+2028/U+2029。
            lines = re.split(r"\r\n|\r|\n", block.decode("utf-8"))
            data = [line[5:].removeprefix(" ") for line in lines if line.startswith("data:")]
            if data:
                events.append("\n".join(data))
        return events


def _stream_sample(raw):
    """有界原始字节证据；hash/长度覆盖全部数据，截断显式标注。"""
    limit = 16384
    return {"bytes": len(raw), "sha256": digest(raw), "truncated": len(raw) > limit,
            "base64": base64.b64encode(raw[:limit]).decode("ascii")}


class TokenCounter:
    """只接受逐输出事件的累计 usage；重复值不累加，不从文本/token_ids 推算。"""

    def __init__(self):
        self.total = 0
        self.updates = 0
        self.observed = False

    def update(self, event):
        if not isinstance(event, dict):
            raise ProtocolError("SSE event must be a JSON object")
        if event.get("error") is not None:
            raise ProtocolError(f"server stream error: {event['error']!r}")
        choices = event.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1 or any(
            not isinstance(choice, dict) or type(choice.get("index", 0)) is not int or choice.get("index", 0) != 0 for choice in choices
        ):
            raise ProtocolError("expected exactly one completion choice")
        for choice in choices:
            if choice.get("delta") or choice.get("finish_reason") not in (None, "length", "stop"):
                raise ProtocolError(f"unsupported/failed completion choice: {choice!r}")
        emitted = any(choice.get("text") or choice.get("token_ids") for choice in choices)
        usage = event.get("usage")
        if usage is None:
            if emitted:
                raise ProtocolError("output event has no continuous usage; refusing chunk counting")
            return
        if not isinstance(usage, dict):
            raise ProtocolError("usage must be a JSON object")
        count = usage.get("completion_tokens")
        if type(count) is not int or count < self.total:
            raise ProtocolError(f"invalid/nonmonotonic cumulative completion_tokens: {count!r}")
        self.total = count
        self.observed = True
        self.updates += 1


def request_payload(row, config):
    if config.backend == "sglang":
        return {"input_ids": row.prompt, "stream": True,
                "sampling_params": {"temperature": 0, "max_new_tokens": row.output_len, "ignore_eos": True}}
    return {"model": config.served_model_name, "prompt": row.prompt, "max_tokens": row.output_len,
            "temperature": 0, "seed": config.seed, "ignore_eos": True, "n": 1, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}


async def _replay(session, base, rows, results, started, deadline, config):
    # 单个顺序 coroutine 保证 C1；trace 到达时间锚定 T0，asap 的计划 offset 全为0。
    loop = asyncio.get_running_loop()
    for row, result in zip(rows, results):
        due = started + result["scheduled_offset_sec"]
        await asyncio.sleep(max(0, due - loop.time()))
        if loop.time() >= deadline:
            return
        counter, decoder = TokenCounter(), SSEDecoder()
        previous = deque(maxlen=2)
        headers = {}
        result.update(status="running", start_offset_sec=loop.time() - started, start_unix_sec=time.time())
        done = False
        try:
            path = "/generate" if config.backend == "sglang" else "/v1/completions"
            async with session.post(base + path, json=request_payload(row, config),
                                    timeout=aiohttp.ClientTimeout(total=None), allow_redirects=False) as response:
                result["http_status"] = response.status
                headers = {key: response.headers[key] for key in ("Content-Type", "Content-Encoding") if key in response.headers}
                if response.status != 200:
                    # 错误头已是确定失败；不能等body到截止后把HTTP失败误记为正常partial。
                    raise ProtocolError(f"HTTP {response.status}")
                if response.content_type != "text/event-stream":
                    raise ProtocolError("completion response must be text/event-stream")
                async for chunk in response.content.iter_any():
                    if loop.time() >= deadline:
                        break
                    events = decoder.feed(chunk)
                    for event_index, data in enumerate(events):
                        if loop.time() >= deadline:
                            break
                        if data == "[DONE]":
                            done = True
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError as exc:
                            result["protocol_evidence"] = {
                                "kind": "malformed_json_event", "headers": headers,
                                "event": _stream_sample(data.encode("utf-8")),
                                "previous": [_stream_sample(item.encode("utf-8")) for item in previous],
                                # 只保存已经收到的后续 event，不为取证继续网络请求。
                                "following_received": [_stream_sample(item.encode("utf-8")) for item in events[event_index + 1:event_index + 3]],
                            }
                            raise ProtocolError(f"malformed SSE JSON: {exc}") from exc
                        previous.append(data)
                        if config.backend == "sglang":
                            if not isinstance(event, dict) or event.get("error") is not None:
                                raise ProtocolError(f"invalid SGLang event: {event!r}")
                            meta = event.get("meta_info")
                            if not isinstance(meta, dict) or "completion_tokens" not in meta:
                                raise ProtocolError("SGLang event missing exact completion_tokens")
                            if type(meta.get("prompt_tokens")) is not int or meta["prompt_tokens"] != row.prompt_len:
                                raise ProtocolError("SGLang prompt_tokens differ from original input IDs")
                            reason = meta.get("finish_reason")
                            if reason is not None and (not isinstance(reason, dict) or reason.get("type") not in {"length", "stop"}):
                                raise ProtocolError(f"SGLang failed finish_reason: {reason!r}")
                            event = {"usage": {"completion_tokens": meta["completion_tokens"]},
                                     "choices": [{"finish_reason": reason["type"] if reason else None}]}
                            done = reason is not None
                        counter.update(event)
                        if counter.total > row.output_len:
                            raise ProtocolError("received output tokens exceed unchanged original budget")
                        for choice in event.get("choices", []):
                            if choice.get("finish_reason") is not None:
                                result["finish_reason"] = choice["finish_reason"]
                    if done:
                        break
                if loop.time() >= deadline:
                    result["status"] = "partial_deadline"
                elif not done:
                    raise ProtocolError("stream ended without [DONE]")
                elif not counter.observed:
                    raise ProtocolError("stream completed without an exact output token counter")
                elif counter.total != row.output_len:
                    raise ProtocolError(f"early completion despite ignore_eos: requested {row.output_len}, got {counter.total}")
                else:
                    result["status"] = "finished"
        except asyncio.CancelledError:
            result["status"] = "partial_deadline" if loop.time() >= deadline else "partial_cancelled"
            raise
        except Exception as exc:
            if "protocol_evidence" not in result:
                result["protocol_evidence"] = {
                    "kind": "transport_error" if isinstance(exc, aiohttp.ClientError) else "stream_protocol_error",
                    "headers": headers, "pending_bytes": _stream_sample(decoder.buffer),
                    "previous": [_stream_sample(item.encode("utf-8")) for item in previous],
                }
            result.update(status="failure", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            result.update(got_output_tokens=counter.total, counter_observed=counter.observed,
                          counter_updates=counter.updates, end_offset_sec=min(loop.time(), deadline) - started,
                          end_unix_sec=time.time())


async def _run_window(endpoint, output_dir, config):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"use a fresh output directory: {output}")
    summary = {"status": "startup_failure", "measured": False, "output_tps": None,
               "actual_received_output_tokens": 0, "actual_elapsed_wall_sec": None,
               "correctness_gate_run": False, "warmup": "none; cold first request included"}
    results = []
    started = deadline = manifest = None
    loop = asyncio.get_running_loop()
    try:
        options = _parse_config(config)
        measurement_protocol = ("c1-fixed-window-token-ids-asap-v1" if options.arrival_mode == "asap"
                                else "c1-fixed-window-token-ids-v1")
        summary.update(arrival_mode=options.arrival_mode, measurement_protocol=measurement_protocol,
                       request_rate="inf" if options.arrival_mode == "asap" else "trace")
        base = _base_url(endpoint)
        rows, metadata, raw_trace = load_workload(options)
        params = {"model": options.served_model_name, "context_length": options.context_length, "concurrency": 1,
                  "duration_sec": options.duration_sec, "protocol": "token-ids-v1", "seed": options.seed,
                  "measurement_protocol": measurement_protocol, "arrival_mode": options.arrival_mode,
                  "request_rate": "inf" if options.arrival_mode == "asap" else "trace",
                  "temperature": 0, "ignore_eos": True, "slowdown_factor": options.slowdown_factor, "num_rounds": 1,
                  "backend": options.backend, "idle_backend": options.idle_backend,
                  "http_connection_reuse": False,
                  "counter_mode": "meta_info.completion_tokens" if options.backend == "sglang" else "usage",
                  "stream_options": None if options.backend == "sglang" else {"include_usage": True, "continuous_usage_stats": True},
                  "return_token_ids": False, "warmup_requests": options.warmup_requests,
                  "warmup_timeout_sec": options.warmup_timeout_sec, "num_prompts": options.num_prompts,
                  "window_expected_engines": options.expected_engines,
                  "stop_when_all_requests_finish": not options.full_duration, "all_requests_required": False}
        for index, row in enumerate(rows):
            trace_offset = (row.timestamp - rows[0].timestamp) / 1000.0
            results.append({"request_index": index, "trace_timestamp_ms": row.timestamp,
                            "trace_offset_sec": trace_offset,
                            "scheduled_offset_sec": trace_offset * options.slowdown_factor if options.arrival_mode == "trace" else 0.0,
                            "requested_prompt_tokens": row.prompt_len, "requested_output_tokens": row.output_len,
                            "prompt_ids_sha256": digest(canonical(row.prompt)),
                            "status": "not_started", "start_offset_sec": None, "end_offset_sec": None,
                            "start_unix_sec": None, "end_unix_sec": None, "got_output_tokens": 0,
                            "counter_mode": params["counter_mode"], "counter_observed": False, "counter_updates": 0})
        summary["trace_total_requests"] = metadata["trace_total_requests"]
        manifest = {**metadata, "endpoint": base, "parameters": params, "parameters_sha256": digest(canonical(params)),
                    "client_sha256": digest(Path(__file__).read_bytes()), "requests": results,
                    "counter_definition": "cumulative completion_tokens in received SSE events",
                    "caveat": "Client-received exact tokens only; server-generated tokens not yet delivered at cutoff are not counted. No text re-tokenization or metrics fallback. No other generation traffic allowed."}
        (output / "trace.original.jsonl").write_bytes(raw_trace)
        write_json(output / "manifest.json", manifest)
        idle = await _wait_for_idle(base, output / "prewindow-idle.json", expected_engines=options.expected_engines, backend=options.idle_backend)
        models = idle["models_response"]
        if options.served_model_name not in {item["id"] for item in models["data"]}:
            raise ValueError(f"{options.served_model_name} absent from /v1/models")
        manifest["models_response"] = models
        # 原时序可恰好撞到服务端5秒keep-alive截止；每条请求新建连接，不重试非幂等POST。
        async with aiohttp.ClientSession(trust_env=False, connector=aiohttp.TCPConnector(force_close=True)) as session:
            if options.warmup_requests:
                # 独立记录预热；不消费正式 trace 行、不复用正式窗口的时钟或计数。
                warmup_row = dataclasses.replace(rows[0], output_len=min(rows[0].output_len, 32))
                warmup_results = [dict(results[0], request_index=index, scheduled_offset_sec=0,
                                       requested_output_tokens=warmup_row.output_len)
                                  for index in range(options.warmup_requests)]
                warmup = {"status": "failed", "requests": warmup_results,
                          "request_count": options.warmup_requests, "wall_time_sec": None}
                summary["warmup"] = manifest["warmup"] = warmup
                warmup_started = loop.time()
                warmup_deadline = warmup_started + options.warmup_timeout_sec
                try:
                    async with asyncio.timeout_at(warmup_deadline):
                        await _replay(session, base, [warmup_row] * options.warmup_requests,
                                      warmup_results, warmup_started, warmup_deadline, options)
                    if any(result["status"] != "finished" for result in warmup_results):
                        raise ProtocolError("warmup requests must finish before measurement")
                    warmup["status"] = "completed"
                finally:
                    warmup["wall_time_sec"] = loop.time() - warmup_started
                manifest["warmup_idle"] = await _wait_for_idle(
                    base, output / "warmup-idle.json", expected_engines=options.expected_engines, backend=options.idle_backend)
            started = loop.time()
            deadline = started + options.duration_sec
            started_wall = time.time()
            manifest.update(start_unix_sec=started_wall, deadline_unix_sec=started_wall + options.duration_sec,
                            start_monotonic_sec=started, deadline_monotonic_sec=deadline)
            for result in results:
                result["window_start_unix_sec"] = started_wall
                result["deadline_unix_sec"] = started_wall + options.duration_sec
            summary.update(measured=True, status="running", start_unix_sec=started_wall,
                           deadline_unix_sec=started_wall + options.duration_sec, counter_mode=params["counter_mode"])
            write_json(output / "manifest.json", manifest)
            timer = asyncio.timeout_at(deadline)
            try:
                async with timer:
                    await _replay(session, base, rows, results, started, deadline, options)
                    if options.full_duration:
                        if options.arrival_mode == "asap" and loop.time() < deadline:
                            raise WorkloadExhausted("ASAP workload exhausted before the fixed window deadline; a larger frozen workload is required")
                        await asyncio.sleep(max(0, deadline - loop.time()))
                if all(r["status"] == "finished" for r in results):
                    summary["status"] = "completed"
                elif loop.time() >= deadline:
                    summary["status"] = "deadline"
                else:
                    raise ProtocolError("replay stopped before deadline with unfinished requests")
            except TimeoutError:
                if not timer.expired():
                    raise
                summary["status"] = "deadline"
            finally:
                ended = loop.time()
                summary["actual_elapsed_wall_sec"] = options.duration_sec if ended >= deadline else ended - started
                summary["end_unix_sec"] = started_wall + summary["actual_elapsed_wall_sec"]
                summary["deadline_dispatch_overshoot_sec"] = max(0, ended - deadline)
    except asyncio.CancelledError:
        summary["status"] = "cancelled" if started is not None else "startup_cancelled"
    except Exception as exc:
        status = "workload_exhausted" if isinstance(exc, WorkloadExhausted) else (
            "measurement_failure" if started is not None else "startup_failure")
        summary.update(status=status, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        summary["actual_received_output_tokens"] = sum(r["got_output_tokens"] for r in results)
        summary["request_counts"] = {status: sum(r["status"] == status for r in results)
                                     for status in ("finished", "partial_deadline", "partial_cancelled", "failure", "not_started", "running")}
        summary["trace_request_count"] = summary["selected_request_count"] = len(results)
        summary["requested_output_tokens_total"] = sum(r["requested_output_tokens"] for r in results)
        # 外部取消、HTTP/解析/计数错误均无可评分 TPS，即使已收到了部分 token。
        summary["counter_valid"] = summary["measured"] and summary["status"] in {"completed", "deadline"}
        if started is not None and summary["actual_elapsed_wall_sec"] is None:
            summary["actual_elapsed_wall_sec"] = min(loop.time(), deadline) - started
            summary["end_unix_sec"] = summary["start_unix_sec"] + summary["actual_elapsed_wall_sec"]
        elapsed = summary["actual_elapsed_wall_sec"]
        if elapsed is not None and elapsed > 0 and summary["counter_valid"]:
            summary["output_tps"] = summary["actual_received_output_tokens"] / elapsed
        summary["throughput_definition"] = "exact output tokens received within fixed window / actual window wall seconds, including in-flight partials, prefill and scheduling; timestamp waits only in trace arrival mode"
        summary["finalized_unix_sec"] = time.time()
        if manifest is None:
            write_json(output / "manifest.json", {"requests": results, "startup_error": summary.get("error", summary["status"])})
        else:
            write_json(output / "manifest.json", manifest)
        with (output / "requests.jsonl").open("w", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        write_json(output / "summary.json", summary)
    return summary


def run_window(endpoint: str, output_dir: Path, config: dict) -> dict:
    """同步 API；返回/落盘原始窗口 schema。失败与外部取消均不能视为有效得分。"""
    return asyncio.run(_run_window(endpoint, output_dir, config))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("config", type=Path, help="full verifier-config.json")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_window(args.endpoint, args.output_dir, config)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if summary["status"] in {"completed", "deadline"} and summary["counter_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
