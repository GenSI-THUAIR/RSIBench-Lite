"""共享 replay client 的真实 HTTP/SSE 错误与 token 计数回归，不加载模型。

参数：无。仅在 loopback 临时端口启动 aiohttp 测试服务；tokenizer 用本地桩。
示例：python3 -m unittest discover -s tests -p test_bench_generation_accounting.py
"""

from __future__ import annotations

from verify.resources import payload_file
import ast
import asyncio
import copy
import hashlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from aiohttp import web


REPO = Path(__file__).resolve().parents[1]


def load_benchmark():
    """加载真实 benchmark 模块，仅隔离这些 HTTP 测试不会调用的 HF tokenizer。"""
    transformers = types.ModuleType("transformers")
    for name in (
        "AutoTokenizer", "PreTrainedTokenizer", "PreTrainedTokenizerBase",
        "PreTrainedTokenizerFast",
    ):
        setattr(transformers, name, type(name, (), {}))
    name = "bench_generation_accounting_under_test"
    spec = importlib.util.spec_from_file_location(
        name, payload_file('vendored_bench_serving.py')
    )
    module = importlib.util.module_from_spec(spec)
    previous = {key: sys.modules.get(key) for key in ("transformers", name)}
    sys.modules.update({"transformers": transformers, name: module})
    try:
        spec.loader.exec_module(module)
    finally:
        # 只恢复两个替身，不能回滚真实 numpy 等本轮新加载的扩展模块。
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


bench = load_benchmark()


def native(text="hello", count=2, finish=None):
    """构造 SGLang 累计 text/usage 帧，可选显式结束原因。"""
    meta = {"completion_tokens": count}
    if finish is not None:
        meta["finish_reason"] = finish
    return {"text": text, "meta_info": meta}


class GenerationAccountingTest(unittest.IsolatedAsyncioTestCase):
    async def request(self, backend, events=(), *, raw=None, status=200, stream=True):
        """发送真实本地 HTTP 请求；分片写 SSE，返回 client 结果和实际请求体。"""
        received = []

        async def handler(request):
            received.append(await request.json())
            if not stream:
                return web.json_response(events[0], status=status)
            response = web.StreamResponse(
                status=status, headers={"Content-Type": "text/event-stream"}
            )
            await response.prepare(request)
            body = raw if raw is not None else b"".join(
                ("data: " + (event if isinstance(event, str) else json.dumps(event))
                 + "\n\n").encode() for event in events
            )
            for offset in range(0, len(body), 7):
                await response.write(body[offset:offset + 7])
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/{path:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            path = "generate" if backend == "sglang" else "v1/completions"
            bench.args = types.SimpleNamespace(
                disable_stream=not stream, disable_ignore_eos=False,
                return_logprob=False,
            )
            request_input = bench.RequestFuncInput(
                prompt="unchanged prompt", api_url=f"http://127.0.0.1:{port}/{path}",
                prompt_len=3, output_len=50, model="test-model", lora_name=None,
                image_data=None, extra_request_body={},
            )
            updates = []
            result = await bench.ASYNC_REQUEST_FUNCS[backend](
                request_input, types.SimpleNamespace(update=updates.append)
            )
            self.assertEqual(updates, [1])
            payload = received[0]
            budget = (payload["sampling_params"]["max_new_tokens"]
                      if backend == "sglang" else payload["max_tokens"])
            self.assertEqual(budget, 50)
            return result, payload
        finally:
            await runner.cleanup()

    def assert_failed(self, result, message=None):
        """失败响应必须有诊断且不能拿请求预算或部分 token 冒充完整成功。"""
        self.assertFalse(result.success)
        self.assertEqual(result.output_len, 0)
        self.assertTrue(result.error)
        if message is not None:
            self.assertIn(message, result.error)

    async def test_http_200_context_error_is_not_success(self):
        for backend in ("sglang", "vllm"):
            with self.subTest(backend=backend):
                result, _ = await self.request(backend, [
                    {"error": {"message": "32862 tokens exceeds context 32768"}},
                    "[DONE]",
                ])
                self.assert_failed(result, "32862")

    async def test_sglang_abort_after_partial_output_is_failure(self):
        result, _ = await self.request("sglang", [native(), native(
            count=2, finish={"type": "abort", "status_code": 400,
                             "message": "context length exceeded"},
        ), "[DONE]"])
        self.assert_failed(result, "abort")

    async def test_sglang_actual_cumulative_count_and_metadata_only_tail(self):
        result, _ = await self.request("sglang", [
            native("first", 2), native("first", 2),
            {"meta_info": {"completion_tokens": 4,
                           "finish_reason": {"type": "length"}}}, "[DONE]",
        ])
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_len, 4)
        self.assertEqual(result.generated_text, "first")
        self.assertGreater(result.ttft, 0)
        self.assertEqual(len(result.itl), 2)

    async def test_openai_usage_only_tail_with_empty_choices(self):
        result, payload = await self.request("vllm", [
            {"choices": [{"text": "short", "finish_reason": None}]},
            {"choices": [{"text": "", "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 3}}, "[DONE]",
        ])
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_len, 3)
        self.assertEqual(result.generated_text, "short")
        self.assertEqual(payload["stream_options"], {"include_usage": True})

    async def test_openai_error_after_partial_output_is_failure(self):
        result, _ = await self.request("vllm", [
            {"choices": [{"text": "partial"}], "usage": {"completion_tokens": 2}},
            {"error": "server failed during generation"}, "[DONE]",
        ])
        self.assert_failed(result, "server failed")

    async def test_empty_or_done_only_stream_has_no_ghost_tokens(self):
        for backend in ("sglang", "vllm"):
            for events in ([], ["[DONE]"]):
                with self.subTest(backend=backend, events=events):
                    result, _ = await self.request(backend, events)
                    self.assert_failed(result, "completion_tokens")

    async def test_text_without_actual_token_count_fails_closed(self):
        for backend, event in (
            ("sglang", {"text": "some text"}),
            ("vllm", {"choices": [{"text": "some text", "finish_reason": "stop"}]}),
        ):
            with self.subTest(backend=backend):
                result, _ = await self.request(backend, [event, "[DONE]"])
                self.assert_failed(result, "completion_tokens")

    async def test_eof_before_finish_is_not_completion(self):
        for backend, event in (
            ("sglang", native()),
            ("vllm", {"choices": [{"text": "partial"}],
                      "usage": {"completion_tokens": 2}}),
        ):
            with self.subTest(backend=backend):
                result, _ = await self.request(backend, [event])
                self.assert_failed(result, "Incomplete generation")

    async def test_explicit_finish_can_complete_without_done_frame(self):
        result, _ = await self.request("sglang", [
            native(finish={"type": "stop"}),
        ])
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_len, 2)

    async def test_typed_sse_error_and_error_finish_reason_after_tokens(self):
        for backend, first in (
            ("sglang", native()),
            ("vllm", {"choices": [{"text": "partial"}],
                      "usage": {"completion_tokens": 2}}),
        ):
            with self.subTest(backend=backend):
                raw = ("data: " + json.dumps(first)
                       + '\n\nevent: error\ndata: {"message":"typed stream failure"}'
                       + "\n\ndata: [DONE]\n\n").encode()
                result, _ = await self.request(backend, raw=raw)
                self.assert_failed(result, "typed stream failure")
        result, _ = await self.request("vllm", [
            {"choices": [{"text": "partial", "finish_reason": "error"}],
             "usage": {"completion_tokens": 2}}, "[DONE]",
        ])
        self.assert_failed(result, "Generation rejected")
        result, _ = await self.request("sglang", [native(finish="abort"), "[DONE]"])
        self.assert_failed(result, "Generation aborted")

    async def test_invalid_or_regressing_count_is_rejected(self):
        for count in (-1, True, 1.5, "2", 1):
            with self.subTest(count=count):
                result, _ = await self.request("sglang", [
                    native(count=2), native(count=count), "[DONE]",
                ])
                self.assert_failed(result, "Invalid completion_tokens")

    async def test_zero_actual_tokens_is_not_budget_success(self):
        for backend, event in (
            ("sglang", native("", 0)),
            ("vllm", {"choices": [], "usage": {"completion_tokens": 0}}),
        ):
            with self.subTest(backend=backend):
                result, _ = await self.request(backend, [event, "[DONE]"])
                self.assert_failed(result, "positive completion_tokens")

    async def test_nonstream_json_uses_actual_count_and_errors(self):
        for backend, event in (
            ("sglang", native("short", 1)),
            ("vllm", {"choices": [{"text": "short"}],
                      "usage": {"completion_tokens": 1}}),
        ):
            with self.subTest(backend=backend):
                result, payload = await self.request(backend, [event], stream=False)
                self.assertTrue(result.success, result.error)
                self.assertEqual(result.output_len, 1)
                self.assertNotIn("stream_options", payload)
                result, _ = await self.request(
                    backend, [{"error": "nonstream error"}], stream=False,
                )
                self.assert_failed(result, "nonstream error")

    async def test_http_error_and_malformed_stream_are_failures(self):
        for backend in ("sglang", "vllm"):
            with self.subTest(backend=backend):
                result, _ = await self.request(backend, raw=b"bad request", status=400)
                self.assert_failed(result, "HTTP 400: bad request")
                result, _ = await self.request(backend, raw=b"data: {broken}\n\n")
                self.assert_failed(result)

    async def test_comments_and_data_without_space(self):
        body = (": heartbeat\n\nevent: message\ndata:" + json.dumps(native())
                + "\n\ndata:[DONE]\n\n").encode()
        result, _ = await self.request("sglang", raw=body)
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_len, 2)

    async def test_all_140_results_retained_but_28_failures_score_zero_tokens(self):
        good, _ = await self.request("sglang", [native(), "[DONE]"])
        bad, _ = await self.request("sglang", [{"error": "context exceeded"}, "[DONE]"])
        outputs = [good] * 112 + [bad] * 28
        tokenizer = types.SimpleNamespace(encode=lambda text, **kwargs: list(text))
        metrics, lengths = bench.calculate_metrics([], outputs, 10.0, tokenizer, "sglang")
        self.assertEqual(len(outputs), 140)
        self.assertEqual(len(lengths), 140)
        self.assertEqual(metrics.completed, 112)
        self.assertEqual(lengths[-28:], [0] * 28)
        self.assertEqual(metrics.total_output, 224)
        self.assertEqual(metrics.output_throughput, 22.4)
        self.assertEqual(metrics.completed / 140, 0.8)


class ToyTokenizer:
    """只提供离线词表/文本计数；稀疏 ID 与特殊 ID 用于检验编译器边界。"""

    def __init__(self, reverse=False, specials=None):
        items = [(f"token-{index}", index * 3) for index in range(64)]
        self.vocab = dict(reversed(items) if reverse else items)
        self.all_special_ids = [0, 3, 21, 189] if specials is None else specials

    def get_vocab(self):
        return self.vocab.copy()

    def encode(self, text, **kwargs):
        return list(text)

    def apply_chat_template(self, messages, **kwargs):
        return "CHAT:" + "\n".join(message["content"] for message in messages)


def mooncake_row(input_len=257, output_len=17, hashes=None, timestamp=0):
    """构造完整 trace 行，默认 hash 覆盖严格按 128 token 分块。"""
    return {
        "input_length": input_len, "output_length": output_len,
        "hash_ids": list(range((input_len + 127) // 128)) if hashes is None else hashes,
        "timestamp": timestamp,
    }


class MooncakeCompilerTest(unittest.TestCase):
    def test_actual_cli_parser_accepts_only_declared_arrival_modes(self):
        tree = ast.parse((payload_file('vendored_bench_serving.py')).read_text())
        main = [node for node in tree.body if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"]
        self.assertEqual(len(main), 1)
        code = compile(ast.Module(body=main, type_ignores=[]), "benchmark-cli", "exec")
        for mode in (None, "trace", "asap", "inf"):
            captured = []
            namespace = {**vars(bench), "__name__": "__main__", "run_benchmark": captured.append}
            argv = ["benchmark", "--dataset-name", "mooncake", "--request-rate", "inf"]
            if mode is not None:
                argv += ["--mooncake-arrival-mode", mode]
            with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
                if mode == "inf":
                    with self.assertRaises(SystemExit) as error:
                        exec(code, namespace)
                    self.assertEqual(error.exception.code, 2)
                    self.assertEqual(captured, [])
                else:
                    exec(code, namespace)
                    self.assertEqual(captured[0].mooncake_arrival_mode, mode)
                    self.assertEqual(captured[0].request_rate, float("inf"))

    def compile_rows(self, records, **kwargs):
        options = dict(backend="sglang", context_length=512, seed=42)
        options.update(kwargs)
        return bench.compile_mooncake_requests(records, ToyTokenizer(), **options)

    def test_exact_blocks_partial_tail_and_shared_prefix(self):
        records = [
            mooncake_row(129, hashes=[11, 22]),
            mooncake_row(256, hashes=[11, 22], timestamp=1),
            mooncake_row(384, hashes=[33, 11, 22], timestamp=2),
        ]
        saved = copy.deepcopy(records)
        rows = self.compile_rows(records)
        self.assertEqual([len(row.prompt) for row in rows], [129, 256, 384])
        self.assertEqual(rows[0].prompt, rows[1].prompt[:129])
        self.assertEqual(rows[1].prompt, rows[2].prompt[128:])
        self.assertNotEqual(rows[2].prompt[:128], rows[2].prompt[128:256])
        self.assertEqual(records, saved)
        allowed = set(ToyTokenizer().get_vocab().values()) - set(ToyTokenizer().all_special_ids)
        self.assertTrue(all(token in allowed and type(token) is int
                            for row in rows for token in row.prompt))

    def test_seed_order_and_vocab_insertion_independence(self):
        records = [mooncake_row(128, hashes=["prefix"], timestamp=10),
                   mooncake_row(128, hashes=["other"], timestamp=20)]
        state = random.getstate()
        reference = self.compile_rows(records)
        self.assertEqual(random.getstate(), state)
        reordered = bench.compile_mooncake_requests(
            records[::-1], ToyTokenizer(reverse=True), backend="vllm",
            context_length=512, seed=42,
        )
        self.assertEqual(reference, reordered)
        self.assertNotEqual(reference[0].prompt, self.compile_rows(records, seed=43)[0].prompt)
        self.assertEqual([row.timestamp for row in reference], [10, 20])

    def test_stable_across_python_hash_seeds(self):
        code = (
            "from test_bench_generation_accounting import bench, ToyTokenizer, mooncake_row; "
            "import hashlib,json; "
            "r=bench.compile_mooncake_requests([mooncake_row()],ToyTokenizer(),"
            "backend='sglang',context_length=512,seed=42); "
            "print(hashlib.sha256(json.dumps(r[0].prompt).encode()).hexdigest())"
        )
        fingerprints = []
        for seed in ("1", "987654"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPATH=str(REPO / "tests"))
            completed = subprocess.run(
                [sys.executable, "-c", code], env=env, capture_output=True,
                text=True, timeout=30, check=True,
            )
            fingerprints.append(completed.stdout.strip())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_official_adapter_forwards_context_flag_and_real_cli_documents_opt_in(self):
        adapter = payload_file('run_official_bench.py')
        with tempfile.TemporaryDirectory(prefix="mooncake-cli-", dir=REPO / ".agents/workdir") as temp:
            script = Path(temp) / "bench.py"
            script.write_text(
                "import argparse,json\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--mooncake-context-length',type=int,default=None)\n"
                "print(json.dumps(vars(p.parse_args())))\n"
            )
            env = dict(os.environ, HARBOR_BENCH_SERVING=str(script), PYTHONDONTWRITEBYTECODE="1")
            for flags, expected in (([], None), (["--mooncake-context-length", "32768"], 32768)):
                result = subprocess.run([sys.executable, str(adapter), *flags],
                                        env=env, text=True, capture_output=True,
                                        timeout=30, check=True)
                self.assertEqual(json.loads(result.stdout)["mooncake_context_length"], expected)
        code = (
            "import runpy,sys,types; from test_bench_generation_accounting import bench; "
            "t=types.ModuleType('transformers'); "
            "[setattr(t,n,getattr(bench,n)) for n in "
            "('AutoTokenizer','PreTrainedTokenizer','PreTrainedTokenizerBase','PreTrainedTokenizerFast')]; "
            "sys.modules['transformers']=t; "
            "sys.argv=[sys.argv[1],'--help']; runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        env = dict(os.environ, HARBOR_BENCH_SERVING=str(payload_file('vendored_bench_serving.py')),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO / "tests"))
        result = subprocess.run([sys.executable, "-c", code, str(adapter)],
                                env=env, text=True, capture_output=True, timeout=30, check=True)
        help_text = " ".join(result.stdout.split())
        self.assertIn("--mooncake-context-length", help_text)
        self.assertIn("omission preserves legacy text prompts", help_text)
        self.assertIn("independently recalibrated", help_text)

    def test_invalid_trace_lengths_hash_coverage_and_timestamp_rejected(self):
        changes = [
            {"input_length": True}, {"input_length": 0}, {"output_length": 0},
            {"output_length": -1}, {"output_length": 1.5}, {"output_length": 256},
            {"hash_ids": [1, 2]}, {"hash_ids": [1, 2, 3, 4]},
            {"hash_ids": [True, 2, 3]}, {"hash_ids": [-1, 2, 3]},
            {"hash_ids": ["", 2, 3]}, {"timestamp": float("nan")},
            {"timestamp": -1}, {"timestamp": True},
        ]
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.compile_rows([mooncake_row() | change])
        with self.assertRaises(ValueError):
            self.compile_rows([])

    def test_unsupported_backends_rounds_context_and_vocab_fail_closed(self):
        for backend in ("vllm-chat", "sglang-oai-chat", "trt", "lmdeploy"):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "no text fallback"):
                self.compile_rows([mooncake_row()], backend=backend)
        for context in (None, 0, -1, True):
            with self.subTest(context=context), self.assertRaises(ValueError):
                self.compile_rows([mooncake_row()], context_length=context)
        with self.assertRaisesRegex(ValueError, "num-rounds=1"):
            self.compile_rows([mooncake_row()], num_rounds=2)
        tokenizer = ToyTokenizer(specials=list(ToyTokenizer().get_vocab().values()))
        with self.assertRaisesRegex(ValueError, "non-special"):
            bench.compile_mooncake_requests([mooncake_row()], tokenizer,
                                           backend="sglang", context_length=512, seed=42)

    def test_lite_trace_keeps_lengths_budgets_and_hash_blocks(self):
        specifications = [
            ("tasks/infer-llada2p1mini-16b", "agentic-trace-32k.jsonl", 32768, 140, 3301157, 57329),
            # 保留两份历史 trace 的完整预算/块共享回归，归档不参与任务发现。
        ]
        allowed = set(ToyTokenizer().get_vocab().values()) - set(ToyTokenizer().all_special_ids)
        for task, name, context, count, input_total, output_total in specifications:
            with self.subTest(trace=name):
                path = REPO / task / "profile" / name
                original_bytes = path.read_bytes()
                records = [json.loads(line) for line in original_bytes.splitlines() if line.strip()]
                before = copy.deepcopy(records)
                rows = self.compile_rows(records, context_length=context)
                self.assertEqual(records, before)
                self.assertEqual(len(rows), count)
                self.assertEqual(sum(row.prompt_len for row in rows), input_total)
                self.assertEqual(sum(row.output_len for row in rows), output_total)
                seen = {}
                for row, record in zip(rows, sorted(records, key=lambda item: item["timestamp"])):
                    self.assertEqual((row.prompt_len, row.output_len, row.timestamp),
                                     (record["input_length"], record["output_length"], record["timestamp"]))
                    self.assertEqual(len(row.prompt), row.prompt_len)
                    self.assertLessEqual(row.prompt_len + row.output_len, context)
                    self.assertTrue(all(token in allowed for token in row.prompt))
                    for index, hash_id in enumerate(record["hash_ids"]):
                        fragment = tuple(row.prompt[index * 128:(index + 1) * 128])
                        self.assertTrue(1 <= len(fragment) <= 128)
                        if hash_id in seen:
                            length = min(len(fragment), len(seen[hash_id]))
                            self.assertEqual(fragment[:length], seen[hash_id][:length])
                        if len(fragment) > len(seen.get(hash_id, ())):
                            seen[hash_id] = fragment
                self.assertEqual(path.read_bytes(), original_bytes)


class MooncakeProtocolHttpTest(unittest.IsolatedAsyncioTestCase):
    async def run_replay(self, backend, records, context=None, *,
                         prompt_delta=0, report_prompt=True, response_delay=0, **options):
        """跑真实 loopback benchmark 请求；只模拟服务端 usage，不证明真实引擎支持。"""
        self.received = []
        self.active = self.max_active = 0

        async def handler(request):
            payload = await request.json()
            self.received.append((request.path, payload))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(response_delay)
            finally:
                self.active -= 1
            prompt = payload.get("input_ids", payload.get("text", payload.get("prompt")))
            prompt_tokens = len(prompt) + prompt_delta
            if backend in ("sglang", "sglang-native"):
                event = native("reply", 2, {"type": "stop"})
                if report_prompt:
                    event["meta_info"]["prompt_tokens"] = prompt_tokens
                events = [event, "[DONE]"]
            else:
                usage = {"completion_tokens": 2}
                if report_prompt:
                    usage["prompt_tokens"] = prompt_tokens
                events = [{"choices": [{"text": "reply", "finish_reason": "stop"}]},
                          {"choices": [], "usage": usage}, "[DONE]"]
            body = "".join("data: " + (item if isinstance(item, str) else json.dumps(item))
                           + "\n\n" for item in events)
            return web.Response(text=body, content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/{path:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            base_url = f"http://127.0.0.1:{port}"
            endpoint = "/generate" if backend in ("sglang", "sglang-native") else "/v1/completions"
            with tempfile.TemporaryDirectory(prefix="mooncake-client-", dir=REPO / ".agents/workdir") as temp:
                bench.args = types.SimpleNamespace(
                    dataset_name="mooncake", mooncake_context_length=context, seed=42,
                    mooncake_num_rounds=options.get("mooncake_num_rounds", 1),
                    warmup_requests=options.get("warmup_requests", 1), backend=backend,
                    sharegpt_output_len=None, random_input_len=0, random_output_len=0,
                    random_range_ratio=1.0, output_file=str(Path(temp) / "result.jsonl"),
                    output_details=True, disable_stream=False, disable_ignore_eos=False,
                    return_logprob=False,
                )
                kwargs = dict(
                    backend=backend, api_url=base_url + endpoint, base_url=base_url,
                    model_id="test-model", tokenizer=ToyTokenizer(),
                    input_requests=copy.deepcopy(records), request_rate=float("inf"),
                    max_concurrency=4, disable_tqdm=True, lora_names=None,
                    extra_request_body={}, profile=False, mooncake_slowdown_factor=1.0,
                )
                kwargs.update(options)
                # 同 event loop 内不做同步辅助 HTTP；generation 本身走真实 aiohttp。
                info = types.SimpleNamespace(status_code=200, json=lambda: {"internal_states": [{}]})
                with patch.object(bench.requests, "get", return_value=info), \
                     patch.object(bench.time, "sleep"), redirect_stdout(io.StringIO()):
                    result = await bench.benchmark(**kwargs)
                self.assertEqual(json.loads(Path(bench.args.output_file).read_text())["completed"], len(records))
                return result
        finally:
            await runner.cleanup()

    async def test_native_and_completions_send_identical_ids_for_warmup_and_replay(self):
        records = [mooncake_row(129, 17, [11, 22]), mooncake_row(255, 51, [11, 22])]
        expected = bench.compile_mooncake_requests(records, ToyTokenizer(),
                                                  backend="sglang", context_length=512, seed=42)
        for backend in ("sglang", "vllm"):
            with self.subTest(backend=backend):
                result = await self.run_replay(backend, records, 512)
                self.assertEqual(len(self.received), 3)
                prompts, budgets = [], []
                for path, payload in self.received:
                    if backend == "sglang":
                        self.assertEqual(path, "/generate")
                        self.assertNotIn("text", payload)
                        prompts.append(payload["input_ids"])
                        budgets.append(payload["sampling_params"]["max_new_tokens"])
                    else:
                        self.assertEqual(path, "/v1/completions")
                        prompts.append(payload["prompt"])
                        budgets.append(payload["max_tokens"])
                self.assertEqual(prompts, [expected[0].prompt, expected[0].prompt, expected[1].prompt])
                self.assertEqual(budgets, [17, 17, 51])
                self.assertEqual(result["mooncake_prompt_protocol"], "token-ids-v1")
                self.assertEqual(result["mooncake_context_length"], 512)
                self.assertEqual(result["mooncake_seed"], 42)
                self.assertEqual(result["server_input_lens"], [129, 255])
                self.assertEqual(result["request_timestamps_ms"], [0, 0])
                self.assertEqual(result["requested_output_lens"], [17, 51])
                self.assertEqual(result["total_input_tokens"], 384)

    async def test_omitting_context_preserves_legacy_text_warmup_and_requests(self):
        records = [mooncake_row(129, 17, [11, 22])]
        with patch.object(bench, "compile_mooncake_requests", side_effect=AssertionError("implicit protocol switch")):
            result = await self.run_replay("sglang", records)
        self.assertEqual(result["mooncake_prompt_protocol"], "legacy-text-v0")
        self.assertIsNone(result["mooncake_context_length"])
        self.assertEqual(len(self.received), 2)
        warmup, formal = [payload["text"] for _, payload in self.received]
        self.assertEqual(warmup.count(" hi"), 2 * 511)
        self.assertEqual(formal.count(" hi"), 2 * 127)
        self.assertIn("Round 1:", formal)
        self.assertNotIn("server_input_lens", result)

    async def test_warmup_requires_actual_server_prompt_count(self):
        for backend in ("sglang", "vllm"):
            for options in ({"prompt_delta": 1}, {"report_prompt": False}):
                with self.subTest(backend=backend, options=options):
                    with self.assertRaisesRegex(ValueError, "server prompt_tokens"):
                        await self.run_replay(backend, [mooncake_row()], 512, **options)
                    self.assertEqual(len(self.received), 1)

    async def test_all_rows_validated_before_warmup_and_no_overrides(self):
        cases = [
            ([mooncake_row(128, 17), mooncake_row(128, 500)], 512, {}),
            ([mooncake_row()], 0, {}),
            ([mooncake_row()], 512, {"warmup_requests": 0}),
            ([mooncake_row()], 512, {"mooncake_num_rounds": 2}),
            ([mooncake_row()], 512, {"extra_request_body": {"max_tokens": 1}}),
            ([mooncake_row()], 512, {"extra_request_body": {"input_ids": [3]}}),
        ]
        for records, context, options in cases:
            with self.subTest(context=context, options=options):
                with self.assertRaises(ValueError):
                    await self.run_replay("sglang", records, context, **options)
                self.assertEqual(self.received, [])
        with self.assertRaisesRegex(ValueError, "no text fallback"):
            await self.run_replay("vllm-chat", [mooncake_row()], 512)
        self.assertEqual(self.received, [])

    async def test_asap_http_keeps_payloads_warmup_and_concurrency(self):
        records = [mooncake_row(129 + index, 17 + index, [11, 22], timestamp=index * 86400000)
                   for index in range(8)]
        immediate = [{**record, "timestamp": 0} for record in records]
        for backend in ("sglang", "vllm"):
            for context in (None, 512):
                with self.subTest(backend=backend, context=context):
                    old = await self.run_replay(backend, immediate, context)
                    expected = copy.deepcopy(self.received)
                    self.assertEqual(old["arrival_mode"], "trace")
                    self.assertEqual(old["request_rate"], "trace")
                    for concurrency in (1, 4):
                        result = await asyncio.wait_for(self.run_replay(
                            backend, records, context, mooncake_arrival_mode="asap",
                            max_concurrency=concurrency, response_delay=.02), timeout=3)
                        self.assertEqual(self.received, expected)
                        self.assertEqual(self.max_active, concurrency)
                        self.assertEqual(result["arrival_mode"], "asap")
                        self.assertEqual(result["request_rate"], "inf")
                        self.assertEqual(result["measurement_protocol"], "mooncake-full-replay-asap-v1")
                        if context is not None:
                            self.assertEqual(result["request_timestamps_ms"], [row["timestamp"] for row in records])

    async def test_asap_invalid_and_conflicting_modes_fail_before_warmup(self):
        cases = [{"mooncake_arrival_mode": value} for value in ("inf", "ASAP", "", True, [], {})]
        cases += [
            {"mooncake_arrival_mode": "asap", "use_trace_timestamps": True},
            {"mooncake_arrival_mode": "asap", "request_rate": 2},
            {"mooncake_arrival_mode": "asap", "request_rate": float("nan")},
        ]
        for context in (None, 512):
            for options in cases:
                with self.subTest(context=context, options=options), self.assertRaises(ValueError):
                    await self.run_replay("sglang", [mooncake_row()], context, **options)
                self.assertEqual(self.received, [])

    async def test_both_generators_asap_skip_wait_without_rewriting_workload(self):
        records = [mooncake_row(1, 2, timestamp=stamp) for stamp in (1000, 1100, 1300)]
        rows = bench.compile_mooncake_requests(records, ToyTokenizer(), backend="sglang", context_length=128, seed=42)
        for generator in (
            lambda mode: bench.get_mooncake_token_requests_over_time(rows, 2.0, mode),
            lambda mode: bench.get_mooncake_request_over_time(copy.deepcopy(records), ToyTokenizer(), 2.0, 1, mode),
        ):
            clock, sleeps = [0.0], []
            async def sleep(delay):
                sleeps.append(delay)
                clock[0] += delay
            with patch.object(bench.time, "perf_counter", side_effect=lambda: clock[0]), \
                 patch.object(bench.asyncio, "sleep", side_effect=sleep):
                original = [row async for row in generator("trace")]
                self.assertEqual(len(sleeps), 2)
                sleeps.clear()
                immediate = [row async for row in generator("asap")]
                self.assertEqual(sleeps, [])
                self.assertEqual(immediate, original)

    async def test_timestamp_scheduler_preserves_milliseconds_and_slowdown(self):
        records = [mooncake_row(1, 2, timestamp=timestamp) for timestamp in (1000, 1100, 1300)]
        rows = bench.compile_mooncake_requests(records, ToyTokenizer(),
                                              backend="sglang", context_length=128, seed=42)
        clock = [0.0]
        sleeps = []

        async def sleep(delay):
            sleeps.append(delay)
            clock[0] += delay

        with patch.object(bench.time, "perf_counter", side_effect=lambda: clock[0]), \
             patch.object(bench.asyncio, "sleep", side_effect=sleep):
            emitted = [row async for row in bench.get_mooncake_token_requests_over_time(rows, 2.0)]
        self.assertEqual(emitted, rows)
        self.assertEqual([row.timestamp for row in emitted], [1000, 1100, 1300])
        self.assertEqual(len(sleeps), 2)
        self.assertAlmostEqual(sleeps[0], 0.2)
        self.assertAlmostEqual(sleeps[1], 0.4)


if __name__ == "__main__":
    unittest.main()
