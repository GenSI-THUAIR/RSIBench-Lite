"""native 图文模板的 processor 桩与真实 loopback 请求体回归，不加载模型。

参数：无。32 请求/并发 8/输出预算 256；只证明传输契约，不证明视觉塔实际参与。
示例：PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_bench_multimodal_protocol.py
"""

from __future__ import annotations

from verify.resources import payload_file
import asyncio
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web

from test_bench_generation_accounting import load_benchmark


REPO = Path(__file__).resolve().parents[1]
ADAPTER = payload_file('run_official_bench.py')
spec = importlib.util.spec_from_file_location("mm_bench_adapter", ADAPTER)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

# 保留 R1 镜像 create_mm_data_row 的计数与 raw/template 选择结构；省略无关模型兜底。
# 图像对象及 data URI 都是桩，不将 loopback 成功冒充 JPEG/视觉模型正确性验证。
ROW_SOURCE = '''\
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class DatasetRow:
    prompt: str
    prompt_len: int
    output_len: int
    text_prompt_len: int
    vision_prompt_len: int
    image_data: list

def create_mm_data_row(text_prompt, images, images_base64, output_len, processor, backend):
    content_items = [
        {"type": "image", "image": {"url": image_base64}}
        for image_base64 in images_base64
    ]
    content_items.append({"type": "text", "text": text_prompt})
    prompt_str = processor.apply_chat_template(
        [{"role": "user", "content": content_items}],
        add_generation_prompt=True, tokenize=False,
    )
    prompt_len = processor(
        text=[prompt_str], images=images, padding=False, return_tensors="pt",
    )["input_ids"].numel()
    text_only_prompt = processor.apply_chat_template(
        [{"role": "user", "content": text_prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    text_prompt_len = processor(
        text=[text_only_prompt], padding=False, return_tensors="pt",
    )["input_ids"].numel()
    vision_prompt_len = prompt_len - text_prompt_len
    use_raw_prompt = backend in [
        "sglang", "sglang-oai", "sglang-oai-chat", "vllm", "vllm-chat",
        "lmdeploy", "lmdeploy-chat",
    ]
    return DatasetRow(
        prompt=text_prompt if use_raw_prompt else prompt_str,
        prompt_len=prompt_len, output_len=output_len,
        text_prompt_len=text_prompt_len, vision_prompt_len=vision_prompt_len,
        image_data=images_base64,
    )
'''


class ProcessorStub:
    """记录未展开模板；仅计数时模拟每图 2042 个视觉 token，不修改发送字符串。"""

    image_token = "<|vision_start|><|image_pad|><|vision_end|>"

    def __init__(self):
        self.counted = []

    def apply_chat_template(self, messages, **kwargs):
        content = messages[0]["content"]
        if isinstance(content, list):
            content = "".join(
                self.image_token if item["type"] == "image" else item["text"]
                for item in content
            )
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, *, text, images=None, **kwargs):
        images = images or []
        self.counted.append((text[0], images))
        if text[0].count(self.image_token) != len(images):
            raise ValueError("image placeholders must match image objects")
        length = len(text[0]) + len(images) * (2042 - len(self.image_token))
        return {"input_ids": types.SimpleNamespace(numel=lambda: length)}


def load_rows(*, corrected=True):
    """编译 fixture，实际调用 adapter 修正，并保留 dataclass 的模块注册语义。"""
    with tempfile.TemporaryDirectory(prefix="mm-protocol-", dir=REPO / ".agents/workdir") as temp:
        script = Path(temp) / "bench.py"
        script.write_text(ROW_SOURCE)
        code = (adapter.native_image_benchmark_code(script) if corrected
                else compile(ROW_SOURCE, str(script), "exec"))
        module = types.ModuleType("mm_rows_fixture")
        with patch.dict(sys.modules, {module.__name__: module}):
            exec(code, module.__dict__)
        return module


def make_row(module, processor, *, backend="sglang", image_count=1, text="raw text"):
    """用可辨别的附件桩构建 row；不生成或变更真实 benchmark 的图片。"""
    images = [object() for _ in range(image_count)]
    data = [f"data:image/jpeg;base64,fixture-{index}" for index in range(image_count)]
    return module.create_mm_data_row(text, images, data, 256, processor, backend)


class MultimodalTemplateTest(unittest.TestCase):
    def test_raw_bug_reproduced_then_template_and_image_count_agree(self):
        original, corrected = load_rows(corrected=False), load_rows()
        for count in (1, 2):
            with self.subTest(images=count):
                before_processor, processor = ProcessorStub(), ProcessorStub()
                before = make_row(original, before_processor, image_count=count)
                row = make_row(corrected, processor, image_count=count)
                self.assertEqual(before.prompt, "raw text")
                self.assertNotEqual(before.prompt, before_processor.counted[0][0])
                self.assertEqual(row.prompt, processor.counted[0][0])
                self.assertEqual(row.prompt.count(processor.image_token), count)
                self.assertEqual(row.prompt.count("<|image_pad|>"), count)
                self.assertEqual(len(row.image_data), count)
                self.assertEqual(row.prompt_len, before.prompt_len)
                self.assertEqual(row.vision_prompt_len, before.vision_prompt_len)
                self.assertEqual(row.output_len, 256)
                self.assertGreater(row.prompt_len, len(row.prompt))
                self.assertEqual(row.image_data, before.image_data)
                self.assertEqual(row.prompt.count("<|im_start|>assistant"), 1)

    def test_other_backends_keep_exact_original_rows(self):
        original, corrected = load_rows(corrected=False), load_rows()
        for backend in ("sglang-oai", "sglang-oai-chat", "vllm", "vllm-chat",
                        "lmdeploy", "lmdeploy-chat", "other-templated-backend"):
            with self.subTest(backend=backend):
                before = make_row(original, ProcessorStub(), backend=backend)
                after = make_row(corrected, ProcessorStub(), backend=backend)
                self.assertEqual(dataclasses.asdict(before), dataclasses.asdict(after))

    def test_unknown_source_shape_fails_before_execution(self):
        variants = (
            "raise AssertionError('script executed before validation')\n",
            ROW_SOURCE.replace("prompt=text_prompt if use_raw_prompt else prompt_str", "prompt=text_prompt"),
            ROW_SOURCE + ROW_SOURCE,
        )
        with tempfile.TemporaryDirectory(prefix="mm-shape-", dir=REPO / ".agents/workdir") as temp:
            script = Path(temp) / "bench.py"
            for source in variants:
                script.write_text(source)
                with self.assertRaisesRegex(RuntimeError, "Unsupported native image"):
                    adapter.native_image_benchmark_code(script)

    def test_cli_loads_selected_image_script_without_modifying_it(self):
        trailer = '''
if __name__ == "__main__":
    import argparse, json, sys
    from test_bench_multimodal_protocol import ProcessorStub
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--dataset-name")
    options, extra = parser.parse_known_args()
    row = create_mm_data_row("raw text", [object()], ["same-image"], 256,
                             ProcessorStub(), options.backend)
    print(json.dumps({"prompt": row.prompt, "argv": sys.argv[1:],
                      "file": __file__, "image_data": row.image_data}))
'''
        with tempfile.TemporaryDirectory(prefix="mm-cli-", dir=REPO / ".agents/workdir") as temp:
            script = Path(temp) / "installed_bench.py"
            source = ROW_SOURCE + trailer
            script.write_text(source)
            env = dict(os.environ, HARBOR_BENCH_SERVING=str(script),
                       PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join((str(REPO / "tests"), str(REPO))))
            workload = ["--num-prompts", "32", "--max-concurrency", "8",
                        "--random-input-len", "2048", "--random-output-len", "256",
                        "--image-count", "1", "--image-resolution", "1080p",
                        "--image-format", "jpeg", "--image-content", "random"]
            for flags, patched in (
                (["--backend", "sglang", "--dataset-name", "image"], True),
                (["--dataset-name=image"], True),
                (["--backend=vllm", "--dataset-name=image"], False),
                (["--backend=sglang", "--dataset-name=mooncake"], False),
            ):
                with self.subTest(flags=flags):
                    result = subprocess.run([sys.executable, str(ADAPTER), *flags, *workload],
                                            env=env, capture_output=True, text=True,
                                            timeout=30, check=True)
                    data = json.loads(result.stdout)
                    self.assertEqual(data["prompt"].count(ProcessorStub.image_token), int(patched))
                    self.assertEqual(data["argv"], flags + workload)
                    self.assertEqual(data["file"], str(script))
                    self.assertEqual(data["image_data"], ["same-image"])
                    self.assertEqual(script.read_text(), source)


class MultimodalLoopbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_32_native_requests_send_counted_unexpanded_template(self):
        """真实 HTTP/SSE，processor/图像用桩；不证明真实服务已执行视觉推理。"""
        bench = load_benchmark()
        module = load_rows()
        processors = [ProcessorStub() for _ in range(32)]
        rows = [make_row(module, processor, text=f"raw text {index}")
                for index, processor in enumerate(processors)]
        received = []
        active, peak = 0, 0

        async def handler(request):
            nonlocal active, peak
            received.append(await request.json())
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            event = {"text": "fixture response", "meta_info": {
                "completion_tokens": 256, "finish_reason": {"type": "length"},
            }}
            return web.Response(text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
                                content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/generate", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            bench.args = types.SimpleNamespace(disable_stream=False, disable_ignore_eos=False,
                                              return_logprob=False)
            semaphore = asyncio.Semaphore(8)

            async def send(row):
                request = bench.RequestFuncInput(
                    prompt=row.prompt, prompt_len=row.prompt_len, output_len=row.output_len,
                    api_url=f"http://127.0.0.1:{port}/generate", model="processor-stub",
                    lora_name=None, image_data=row.image_data, extra_request_body={},
                )
                async with semaphore:
                    return await bench.ASYNC_REQUEST_FUNCS["sglang"](request)

            results = await asyncio.gather(*(send(row) for row in rows))
            self.assertEqual(len(received), 32)
            self.assertTrue(all(result.success for result in results))
            self.assertLessEqual(peak, 8)
            self.assertGreater(peak, 1)
            self.assertEqual({payload["text"] for payload in received},
                             {processor.counted[0][0] for processor in processors})
            for payload in received:
                self.assertNotIn("input_ids", payload)
                self.assertEqual(payload["text"].count(ProcessorStub.image_token), 1)
                self.assertEqual(payload["text"].count("<|image_pad|>"), 1)
                self.assertEqual(payload["image_data"], rows[0].image_data)
                self.assertEqual(payload["sampling_params"]["max_new_tokens"], 256)
                self.assertTrue(payload["sampling_params"]["ignore_eos"])
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    unittest.main()
