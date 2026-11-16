"""验证 opt-in chat 预算与完整 final 门，保留历史默认协议。

参数：无；示例：python -m unittest discover -s tests -p test_gsm8k_eval.py
包含真实本地 HTTP 请求/并发评测，不以模型或 GPU fixture 冒充实机标定。
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from verify.resources import payload_module
gsm = payload_module("gsm8k_eval.py")

WORKDIR = Path(__file__).resolve().parents[1] / ".agents/workdir"


def response(content="#### 42", finish="stop", **message_fields):
    """构造 chat 响应；输入为 content/结束原因/附加消息字段，返回 JSON 对象。"""
    return {
        "choices": [{"finish_reason": finish, "message": {"content": content, **message_fields}}],
        "usage": {"completion_tokens": 123},
    }


class ConfigTests(unittest.TestCase):
    """新键仅允许 chat，省略保持 1024/非 strict。"""

    def load(self, **gate):
        """把给定 gate 写入本树临时文件并经真实配置读取器返回。"""
        WORKDIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=WORKDIR) as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"served_model_name": "fixture", "gate": {"kind": "gsm8k-openai-chat", "stop_tokens": [], **gate}}))
            return gsm.load_eval_config(path)

    def test_defaults_and_explicit_options(self):
        """默认口径不变，显式预算与 strict 原值传递。"""
        default = self.load()
        self.assertEqual(default["max_tokens"], 1024)
        self.assertIs(default["strict_final_answer"], False)
        self.assertIsNone(default["reasoning_effort"])
        configured = self.load(max_tokens=32768, strict_final_answer=True, reasoning_effort="custom-max")
        self.assertEqual(configured["reasoning_effort"], "custom-max")
        self.assertEqual(configured["max_tokens"], 32768)
        self.assertIs(configured["strict_final_answer"], True)

    def test_invalid_options_fail_fast(self):
        """拒绝非正整数预算和非布尔 strict，不把 bool 当 int。"""
        for value in (0, -1, 1.5, "1024", True, None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.load(max_tokens=value)
        for value in (0, 1, "true", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.load(strict_final_answer=value)
        for value in (None, "", " \t\n", 0, 0.99, False, []):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.load(reasoning_effort=value)

    def test_sglang_rejects_even_disabled_chat_options(self):
        """SGLang 显式配置 chat 专用键即报错，不静默忽略。"""
        self.load(kind="gsm8k-sglang-generate")
        for options in ({"max_tokens": 1024}, {"strict_final_answer": False}, {"reasoning_effort": "max"}, {"reasoning_effort": None}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.load(kind="gsm8k-sglang-generate", **options)


class FinalAnswerTests(unittest.TestCase):
    """完整 final 与推理、截断、工具输出之间有明确边界。"""

    def test_strict_invalid_reasons(self):
        """错误 final 即使含正确数字也必须拒绝，并保留具体原因。"""
        cases = [
            (response(None, reasoning_content="#### 42"), "empty_final"),
            (response(None, reasoning="#### 42"), "empty_final"),
            (response("   "), "empty_final"),
            (response(42), "empty_final"),
            (response("#### 42", "length"), "truncated_by_length"),
            (response(None, "length", reasoning="#### 42"), "truncated_by_length"),
            (response("#### 42", None), "non_stop_finish"),
            (response("#### 42", "tool_calls"), "non_stop_finish"),
            (response("#### 42", tool_calls=[{"id": "call"}]), "tool_call"),
            (response("#### 42", function_call={"name": "tool"}), "tool_call"),
            (response("I cannot solve this."), "unparsed_final"),
        ]
        missing_finish = response()
        del missing_finish["choices"][0]["finish_reason"]
        cases.append((missing_finish, "non_stop_finish"))
        for payload, expected_reason in cases:
            with self.subTest(reason=expected_reason, payload=payload):
                diagnostics = {}
                with patch.object(gsm, "_post_json", return_value=payload):
                    prediction, tokens = gsm.generate_openai_chat("http://fixture", "question", "fixture", strict_final_answer=True, diagnostics=diagnostics)
                self.assertEqual((prediction, tokens), (gsm.INVALID_ANSWER, 123))
                self.assertEqual(diagnostics, {"final_answer_status": expected_reason})

    def test_strict_accepts_real_content_with_existing_extractor(self):
        """只从 final 抽答案；不额外引入严格 markdown/最后一行格式门。"""
        for final in ("#### 42", "The answer is 42.", "**#### 42**"):
            with self.subTest(final=final), patch.object(gsm, "_post_json", return_value=response(final, reasoning="#### 999", reasoning_content="#### 998", tool_calls=[])):
                diagnostics = {}
                self.assertEqual(gsm.generate_openai_chat("http://fixture", "q", "fixture", strict_final_answer=True, diagnostics=diagnostics), (42, 123))
                self.assertEqual(diagnostics["final_answer_status"], "valid_final")

    def test_legacy_chat_request_and_fallback_unchanged(self):
        """默认三位置参数调用保留旧 reasoning_content 与截断处理。"""
        with patch.object(gsm, "_post_json", return_value=response(None, "length", reasoning_content="#### 42")) as post:
            self.assertEqual(gsm.generate_openai_chat("http://fixture", "q", "fixture"), (42, 123))
        self.assertEqual(post.call_args.args[1], {
            "model": "fixture",
            "messages": [{"role": "system", "content": gsm.CHAT_SYSTEM_PROMPT}, {"role": "user", "content": "q"}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 1024,
            "stream": False,
        })
        with patch.object(gsm, "_post_json", return_value=response(None, reasoning="#### 42")):
            self.assertEqual(gsm.generate_openai_chat("http://fixture", "q", "fixture")[0], gsm.INVALID_ANSWER)

    def test_effort_is_forwarded_without_model_specific_mapping(self):
        """字符串原样透传，不把max等值改写为模型专属数字。"""
        for effort in ("max", "custom-effort", " high "):
            with self.subTest(effort=effort), patch.object(gsm, "_post_json", return_value=response()) as post:
                result = gsm.generate_openai_chat("http://fixture", "q", "fixture", reasoning_effort=effort, strict_final_answer=True)
            self.assertEqual(result, (42, 123))
            self.assertEqual(post.call_args.args[1]["reasoning_effort"], effort)

    def test_sglang_request_unchanged(self):
        """SGLang 的 stop/512-token 口径保持不变。"""
        with patch.object(gsm, "_post_json", return_value={"text": "#### 42", "meta_info": {"completion_tokens": 5}}) as post:
            self.assertEqual(gsm.generate_sglang("http://fixture", "prompt", ["stop"]), (42, 5))
        self.assertEqual(post.call_args.args[1]["sampling_params"], {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512, "stop": ["stop"]})


class HttpEvaluationTests(unittest.TestCase):
    """真实本地 HTTP 端到端覆盖配置读取、预算、并发、原因汇总。"""

    def test_http_evaluate_strict_and_legacy(self):
        """同一响应池下 strict 拒绝不完整答案，默认旧协议指标保持。"""
        payloads = {
            "valid": response("#### 42"),
            "empty": response(None, reasoning_content="#### 42"),
            "length": response("#### 42", "length"),
            "unparsed": response("No answer."),
        }
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                """接收真实 JSON POST，按问题返回固定响应或 503。"""
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body))
                question = body["messages"][-1]["content"]
                if question == "network":
                    self.send_error(503)
                    return
                raw = json.dumps(payloads[question]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format, *args):
                """关闭 fixture HTTP access log，不影响响应处理。"""

        WORKDIR.mkdir(parents=True, exist_ok=True)
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory(dir=WORKDIR) as directory:
                    data = Path(directory) / "data.jsonl"
                    data.write_text("".join(json.dumps({"question": question, "answer": "#### 42"}) + "\n" for question in [*payloads, "network"]))
                    endpoint = f"http://127.0.0.1:{server.server_port}"
                    base = {"kind": "gsm8k-openai-chat", "served_model_name": "fixture", "num_questions": 5, "stop_tokens": []}
                    with patch.dict("os.environ", {"GSM8K_SMOKE_QUESTIONS": ""}):
                        strict = gsm.evaluate(endpoint, data, {**base, "max_tokens": 32768, "strict_final_answer": True, "reasoning_effort": "max"})
                        legacy = gsm.evaluate(endpoint, data, base)
            finally:
                server.shutdown()
                thread.join(timeout=5)
        self.assertEqual(strict["accuracy"], 0.2)
        self.assertEqual(strict["invalid_rate"], 0.8)
        self.assertEqual(strict["final_answer_diagnostics"], {"valid_final": 1, "empty_final": 1, "truncated_by_length": 1, "unparsed_final": 1, "request_error": 1})
        self.assertEqual(legacy["accuracy"], 0.6)
        self.assertEqual(legacy["invalid_rate"], 0.4)
        self.assertNotIn("final_answer_diagnostics", legacy)
        self.assertEqual(len(requests), 10)
        self.assertTrue(all(path == "/v1/chat/completions" for path, _ in requests))
        self.assertTrue(all(body["max_tokens"] == 32768 for _, body in requests[:5]))
        self.assertTrue(all(body["max_tokens"] == 1024 for _, body in requests[5:]))
        self.assertTrue(all(body["reasoning_effort"] == "max" for _, body in requests[:5]))
        self.assertTrue(all("reasoning_effort" not in body for _, body in requests[5:]))
        self.assertGreater(strict["output_throughput"], 0)


if __name__ == "__main__":
    unittest.main()
