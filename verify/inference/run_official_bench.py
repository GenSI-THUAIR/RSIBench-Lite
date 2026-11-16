"""在 CPU job 中执行官方 bench_serving，并修正 native SGLang 的图文模板传输。

参数：全部命令行参数直接传给 bench_serving;脚本位置由环境变量
HARBOR_BENCH_SERVING 指定(task 的 verifier-config.json 的
bench_serving.script_path,由 clean_benchmark.py 注入)。adapter 提供未使用的
PD 常量，避免 eager import 初始化 Triton 驱动；仅在 sglang + image 路径把
create_mm_data_row 的发送文本改为其计数所用的未展开图文模板，不修改镜像文件。
其他 backend/dataset（含 mooncake）仍原样执行。

示例：
    HARBOR_BENCH_SERVING=/logs/verifier/vendored_bench_serving.py \
        python3 run_official_bench.py --backend vllm --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import ast
import os
import runpy
import sys
from pathlib import Path
from types import CodeType, ModuleType


def native_image_benchmark_code(path: Path) -> CodeType:
    """只修正官方图文 row 的 native prompt 选择；实现漂移时拒绝静默计分。"""
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_mm_data_row"
    ]
    prompts = [
        keyword
        for function in functions
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "DatasetRow"
        for keyword in node.keywords if keyword.arg == "prompt"
    ]
    expected = ast.parse(
        "text_prompt if use_raw_prompt else prompt_str", mode="eval"
    ).body
    if len(functions) != 1 or len(prompts) != 1 or ast.dump(prompts[0].value) != ast.dump(expected):
        raise RuntimeError(
            "Unsupported native image benchmark: expected create_mm_data_row "
            "to select raw/template prompt; review the installed script before scoring"
        )
    # 传的是 apply_chat_template 的字符串，不是 processor 展开后的 input_ids；
    # 服务端仍负责按每张图展开视觉 token，避免重复展开。其他 backend 选择不变。
    prompts[0].value.test = ast.parse(
        "use_raw_prompt and backend != 'sglang'", mode="eval"
    ).body
    return compile(ast.fix_missing_locations(tree), str(path), "exec")


def run_native_image_benchmark(path: Path) -> None:
    """在独立 __main__ 命名空间执行内存修正，保留脚本文件名及 dataclass 语义。"""
    code = native_image_benchmark_code(path)
    module = ModuleType("__main__")
    module.__file__ = str(path)
    module.__package__ = ""
    previous_main, previous_argv0 = sys.modules["__main__"], sys.argv[0]
    try:
        sys.modules["__main__"] = module
        sys.argv[0] = str(path)
        exec(code, module.__dict__)
    finally:
        sys.modules["__main__"] = previous_main
        sys.argv[0] = previous_argv0


def main() -> None:
    """注入 PD 常量；仅 native image benchmark 使用图文模板传输修正。"""
    bench_serving_raw = os.environ.get("HARBOR_BENCH_SERVING")
    if not bench_serving_raw:
        raise RuntimeError(
            "HARBOR_BENCH_SERVING 未设置(应为 verifier-config.json 的 "
            "bench_serving.script_path)"
        )
    bench_serving = Path(bench_serving_raw)
    if not bench_serving.is_file():
        raise FileNotFoundError(f"bench_serving 不存在：{bench_serving}")
    disaggregation_utils = ModuleType("sglang.srt.disaggregation.utils")
    disaggregation_utils.FAKE_BOOTSTRAP_HOST = "2.2.2.2"
    sys.modules[disaggregation_utils.__name__] = disaggregation_utils
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--dataset-name")
    options, _ = parser.parse_known_args()
    if options.backend == "sglang" and options.dataset_name == "image":
        run_native_image_benchmark(bench_serving)
    else:
        runpy.run_path(str(bench_serving), run_name="__main__")


if __name__ == "__main__":
    main()
