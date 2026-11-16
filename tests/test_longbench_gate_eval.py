"""verify/text/longbench_eval.py(gate 口径)的单元测试。

参数：无。用桩 HTTP 服务模拟 /v1/chat/completions,验证官方模板渲染、
答案抽取规则、无效/截断统计、并发与配置校验。

示例：
  PYTHONPATH=. .venv/bin/python3 -m unittest tests.test_longbench_gate_eval
"""

from __future__ import annotations

from verify.resources import payload_file
import importlib.util
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

SCRIPT = payload_file('longbench_eval.py')


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("longbench_eval", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(index: int, answer: str = "A", length: str = "short") -> dict[str, object]:
    return {
        "index": index,
        "_id": f"id{index}",
        "domain": "Single-Document QA",
        "sub_domain": "Academic",
        "difficulty": "easy",
        "length": length,
        "question": f"q{index}",
        "choices": {"A": " w ", "B": "x", "C": "y", "D": "z"},
        "answer": answer,
        "context": f"  doc{index}  ",
    }


class PromptAndExtractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_prompt_follows_official_template(self) -> None:
        prompt = self.module.format_prompt(sample(3))
        self.assertTrue(prompt.startswith("Please read the following text"))
        self.assertIn("<text>\ndoc3\n</text>", prompt)
        self.assertIn("What is the correct answer to this question: q3\nChoices:\n(A) w\n(B) x\n(C) y\n(D) z\n", prompt)
        self.assertTrue(prompt.endswith('"The correct answer is (insert answer here)".'))

    def test_extract_official_patterns_and_fallbacks(self) -> None:
        extract = self.module.extract_answer
        self.assertEqual(extract("The correct answer is (B)"), "B")
        self.assertEqual(extract("The correct answer is C."), "C")
        self.assertEqual(extract("**The correct answer is (D)**"), "D")
        self.assertEqual(extract("The correct answer is (A)\n...wait, The correct answer is (B)"), "B")
        self.assertEqual(extract("<think>The correct answer is (A)</think>The correct answer is (C)"), "C")
        self.assertEqual(extract(r"so \boxed{D}"), "D")
        self.assertEqual(extract("blah\nB\n"), "B")
        self.assertIsNone(extract("no letter here"))
        self.assertIsNone(extract("the correct answer is (a)"))

    def test_config_validation(self) -> None:
        module = self.module
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "verifier-config.json"
            base = {"gate": {"kind": "longbench-v2-openai-chat"}, "served_model_name": "m"}
            path.write_text(json.dumps(base))
            config = module.load_eval_config(path)
            self.assertEqual(
                (config["max_tokens"], config["parallel"], config["num_questions"], config["reasoning_effort"]),
                (module.DEFAULT_MAX_TOKENS, module.DEFAULT_PARALLEL, None, None),
            )
            base["gate"].update(max_tokens=64, max_concurrency=2, num_questions=1, reasoning_effort="low")
            path.write_text(json.dumps(base))
            config = module.load_eval_config(path)
            self.assertEqual((config["max_tokens"], config["parallel"], config["num_questions"]), (64, 2, 1))
            for bad in ({"kind": "gsm8k-openai-chat"}, {"kind": "longbench-v2-openai-chat", "num_questions": 0},
                        {"kind": "longbench-v2-openai-chat", "max_concurrency": True},
                        {"kind": "longbench-v2-openai-chat", "max_tokens": -1},
                        {"kind": "longbench-v2-openai-chat", "reasoning_effort": ""}):
                path.write_text(json.dumps({"gate": bad, "served_model_name": "m"}))
                with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                    module.load_eval_config(path)


class EvaluateStubServerTest(unittest.TestCase):
    def test_accuracy_invalid_truncated_and_buckets(self) -> None:
        module = load_module()
        # 8 题:index 0 回无法解析文本(无效),index 1 被 max_tokens 截断且答错,
        # index 2 只在 reasoning_content 里给出正确答案,其余 content 答对。
        samples = [sample(i, length="short" if i < 4 else "medium") for i in range(8)]
        seen_bodies: list[dict[str, object]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                seen_bodies.append(body)
                prompt = body["messages"][0]["content"]
                question = prompt.split("What is the correct answer to this question: ", 1)[1].split("\n", 1)[0]
                message: dict[str, object] = {"content": "The correct answer is (A)"}
                finish = "stop"
                if question == "q0":
                    message = {"content": "no idea"}
                elif question == "q1":
                    message = {"content": "The correct answer is (B)"}
                    finish = "length"
                elif question == "q2":
                    message = {"content": "", "reasoning_content": "hmm The correct answer is (A)"}
                payload = {
                    "choices": [{"message": message, "finish_reason": finish}],
                    "usage": {"completion_tokens": 5},
                }
                encoded = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args: object) -> None:  # 静默桩服务日志
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                data = Path(tmp) / "lb.jsonl"
                data.write_text("".join(json.dumps(s) + "\n" for s in samples))
                result = module.evaluate(
                    f"http://127.0.0.1:{server.server_port}",
                    data,
                    {"served_model_name": "m", "max_tokens": 32, "parallel": 3, "reasoning_effort": "high"},
                )
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(result["num_questions"], 8)
        self.assertEqual(result["parallel"], 3)
        self.assertEqual(result["max_tokens"], 32)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["invalid_rate"], 0.125)
        self.assertEqual(result["truncated_rate"], 0.125)
        self.assertEqual(result["accuracy_by_length"], {"medium": 1.0, "short": 0.5})
        self.assertEqual(result["invalid_reasons"], {"no_answer": 1})
        self.assertEqual(len(result["invalid_samples"]), 1)
        self.assertIn("no_answer(finish=stop): no idea", result["invalid_samples"][0])
        self.assertGreater(result["output_throughput"], 0)
        self.assertEqual(len(seen_bodies), 8)
        self.assertEqual(seen_bodies[0]["temperature"], 0)
        self.assertEqual(seen_bodies[0]["max_tokens"], 32)
        self.assertEqual(seen_bodies[0]["reasoning_effort"], "high")
        self.assertEqual(seen_bodies[0]["model"], "m")

    def test_unicode_line_separators_inside_context_do_not_split_records(self) -> None:
        # LongBench 原文含 U+2028/U+2029/U+0085 等字符,ensure_ascii=False 写出时保持原样;
        # 读取只能按 \n 切分,否则一条记录会被截成非法 JSON。
        module = load_module()
        rows = [dict(sample(i), context=f"a\u2028b\u2029c\x85d\x0c{i}") for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "lb.jsonl"
            data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "num_questions"):
                module.evaluate("http://127.0.0.1:9", data, {"served_model_name": "m", "num_questions": 4})

    def test_request_errors_are_counted_with_reason(self) -> None:
        module = load_module()
        module.REQUEST_ATTEMPTS = 1
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "lb.jsonl"
            data.write_text("".join(json.dumps(sample(i)) + "\n" for i in range(2)))
            # 无人监听的端口:每题连接被拒,按 request_error 计无效。
            result = module.evaluate("http://127.0.0.1:9", data, {"served_model_name": "m", "parallel": 2})
        self.assertEqual(result["invalid_rate"], 1.0)
        self.assertEqual(result["invalid_reasons"], {"request_error": 2})
        self.assertTrue(result["invalid_samples"][0].startswith("request_error: URLError"))

    def test_num_questions_slices_prefix(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "lb.jsonl"
            data.write_text("".join(json.dumps(sample(i)) + "\n" for i in range(3)))
            with self.assertRaisesRegex(ValueError, "num_questions"):
                module.evaluate("http://127.0.0.1:9", data, {"served_model_name": "m", "num_questions": 4})
            (Path(tmp) / "empty.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "为空"):
                module.evaluate("http://127.0.0.1:9", Path(tmp) / "empty.jsonl", {"served_model_name": "m"})


if __name__ == "__main__":
    unittest.main()
