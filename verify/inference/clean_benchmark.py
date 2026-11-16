"""在未打 patch 的干净 job 中评测远端被测服务。

参数：endpoint 为 patched Serve Job 地址；data_path 为 gate 数据(GSM8K/MMMU
JSONL、参考 MP4 或参考观测/动作 npz，由 config 的 gate.kind 决定 evaluator)；output_dir 用于保存
gate、性能原始输出和汇总结果；config_path 为 task 的 verifier-config.json。

示例：
    python3 clean_benchmark.py \
        'http://[2605:340::1]:8000' /mnt/rsibench/rsibench/gsm8k_test.jsonl \
        /logs/verifier /logs/verifier/verifier-config.json
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import gpqa_eval
import longbench_eval
import gsm8k_eval
import mmmu_eval

BENCH_KEYS = (
    "backend",
    "script_path",
    "dataset_name",
    "random_input_len",
    "random_output_len",
    "random_range_ratio",
    "max_concurrency",
    "num_prompts",
    "seed",
    "extra_args",
)


def load_bench_config(
    config: dict[str, object], config_path: Path
) -> dict[str, object]:
    """读取 bench_serving 后端与协议参数;缺失一律 fail loud。"""
    bench = config["bench_serving"]
    if not isinstance(bench, dict):
        raise TypeError(f"bench_serving 必须是 JSON object：{config_path}")
    missing = [key for key in BENCH_KEYS if key not in bench]
    if missing:
        raise ValueError(f"bench_serving 缺少字段 {missing}：{config_path}")
    mode = bench.get("arrival_mode", "trace")
    if not isinstance(mode, str) or mode not in {"trace", "asap"}:
        raise ValueError("bench_serving.arrival_mode must be trace or asap")
    if "arrival_mode" in bench and bench["dataset_name"] != "mooncake":
        raise ValueError("bench_serving.arrival_mode is only valid for mooncake")
    reserved = {"--mooncake-arrival-mode"}
    if "arrival_mode" in bench:
        reserved.update({"--request-rate", "--use-trace-timestamps"})
    if any(str(arg).partition("=")[0] in reserved for arg in bench["extra_args"]):
        raise ValueError("arrival scheduling must be set by bench_serving.arrival_mode, not extra_args")
    return bench


def _last_json_object(path: Path) -> dict[str, object]:
    """读取 bench_serving JSONL 的最后一个对象。"""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"bench_serving 没有写入结果：{path}")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise TypeError(f"bench_serving 结果不是 JSON object：{path}")
    return value


def _metric(result: dict[str, object], *names: str) -> float:
    """按兼容字段名提取一个有限数值，缺失时返回 -1。"""
    for name in names:
        value = result.get(name)
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                return number
    return -1.0


def _requested_count(bench: dict[str, object], num_prompts: int) -> int:
    """本次实际会发出的请求数——request_success_rate 的正确分母。

    对 random / random-ids 数据集,num_prompts 就是请求数。但 mooncake 回放的
    请求由 trace 文件逐行给出,num_prompts 只是个上界(惯例写 10000),真实
    分母是 trace 行数。踩过:qwen3p8 的 193 条请求全部成功,却因为算成
    193/10000 = 0.0193 被判 FAIL,而它的 baseline 正是按 193/193 标定的。

    trace 路径不可读时退回 num_prompts —— 宁可沿用旧口径,也不要凭空缩小
    分母把失败的跑法算成通过。
    """
    if bench.get("dataset_name") != "mooncake":
        return num_prompts
    extra = [str(arg) for arg in bench.get("extra_args", [])]
    if "--dataset-path" not in extra:
        return num_prompts
    trace = Path(extra[extra.index("--dataset-path") + 1])
    try:
        lines = sum(1 for line in trace.read_text().splitlines() if line.strip())
    except OSError:
        return num_prompts
    if lines == 0:
        return num_prompts
    # trace 比 num_prompts 长时,bench_serving 只发前 num_prompts 条。
    return min(lines, num_prompts)


def _scalar_metrics(result: dict[str, object]) -> dict[str, object]:
    """保留 bench_serving 结果中的全部标量指标作为参考留档。

    丢掉 --output-details 产生的 per-request 数组和长文本;不挑字段名,
    bench_serving 版本新增的聚合指标(E2E/ITL 分位、peak throughput 等)
    自动跟着进 summary。
    """
    return {
        key: value
        for key, value in result.items()
        if isinstance(value, (int, float, str))
        and not isinstance(value, bool)
        and (not isinstance(value, str) or len(value) <= 200)
    }


def _window_performance(
    raw: dict[str, object], bench: dict[str, object]
) -> dict[str, object]:
    """只接收完整有效的限时测量；合法到点截停不是请求失败。"""
    if raw.get("status") not in {"deadline", "completed"}:
        raise ValueError(f"固定窗口未正常结束：{raw.get('status')}；{raw.get('error_type', '')}: {raw.get('error', '')}")
    if raw.get("measured") is not True or raw.get("counter_valid") is not True:
        raise ValueError("固定窗口缺少有效测量或准确 token 计数")
    if raw.get("error") or raw.get("error_type"):
        raise ValueError("固定窗口含错误，不能据 counter_valid 单独放行")
    if type(bench.get("max_concurrency")) is not int or bench["max_concurrency"] != 1:
        raise ValueError("固定窗口仅支持显式 C1")
    duration = bench["window_duration_sec"]
    elapsed = raw.get("actual_elapsed_wall_sec")
    tokens = raw.get("actual_received_output_tokens")
    for name, value in (("window_duration_sec", duration), ("actual_elapsed_wall_sec", elapsed)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"固定窗口 {name} 必须是有限正数")
    if duration > 3600 or elapsed > duration + 1e-6:
        raise ValueError("固定窗口超过声明的测量上限")
    if bench.get("window_full_duration", False) and not math.isclose(elapsed, duration, abs_tol=1e-6, rel_tol=0):
        raise ValueError("完整固定窗口必须覆盖全部声明时长")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("固定窗口输出 token 数必须为非负整数")
    counts = raw.get("request_counts")
    names = {"finished", "partial_deadline", "partial_cancelled", "failure", "not_started", "running"}
    if not isinstance(counts, dict) or set(counts) != names:
        raise ValueError("固定窗口请求状态计数不完整")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("固定窗口请求状态计数必须为非负整数")
    if counts["failure"] or counts["partial_cancelled"] or counts["running"]:
        raise ValueError("固定窗口存在真实失败、提前取消或未收尾请求")
    if counts["partial_deadline"] > 1:
        raise ValueError("C1 固定窗口不允许多个在途截停请求")
    selected = raw.get("selected_request_count", raw.get("trace_request_count"))
    if type(selected) is not int or selected <= 0 or selected != sum(counts.values()):
        raise ValueError("固定窗口请求计数与实际选取数量不一致")
    if selected > int(bench["num_prompts"]):
        raise ValueError("固定窗口超过配置的请求数量上限")
    if raw["status"] == "completed" and counts["finished"] != selected:
        raise ValueError("只有选取的请求全部结束，才允许提前完成窗口")
    if raw["status"] == "deadline" and not math.isclose(elapsed, duration, abs_tol=1e-6, rel_tol=0):
        raise ValueError("到点结束的窗口必须覆盖完整声明时长")
    attempted = counts["finished"] + counts["partial_deadline"]
    if tokens < counts["finished"]:
        raise ValueError("完整请求的正输出预算与 token 总数矛盾")
    if attempted == 0 and tokens != 0:
        raise ValueError("未发起请求却报告了输出 token")
    throughput = tokens / elapsed
    reported = raw.get("output_tps")
    if isinstance(reported, bool) or not isinstance(reported, (int, float)) or not math.isfinite(reported) or not math.isclose(reported, throughput, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("固定窗口 TPS 与 token/墙钟独立重算不一致")
    mode = bench.get("arrival_mode", "trace")
    protocol = "c1-fixed-window-token-ids-asap-v1" if mode == "asap" else "c1-fixed-window-token-ids-v1"
    if raw.get("arrival_mode", "trace") != mode or raw.get("measurement_protocol", "c1-fixed-window-token-ids-v1") != protocol:
        raise ValueError("固定窗口arrival/protocol与配置不一致")
    if mode == "asap" and raw.get("request_rate") != "inf":
        raise ValueError("ASAP窗口缺少无到达节流的request_rate证据")
    return {
        "measurement_protocol": protocol,
        "arrival_mode": mode,
        "request_rate": "inf" if mode == "asap" else "trace",
        "num_prompts": int(bench["num_prompts"]),
        "requests_selected": selected,
        "requests_attempted": attempted,
        "max_concurrency": 1,
        "seed": int(bench["seed"]),
        "completed": counts["finished"],
        "partial_deadline": counts["partial_deadline"],
        "not_started": counts["not_started"],
        "request_failures": counts["failure"],
        # 此处分母仅为窗口内发起的请求，合法截停计为无错误；不是全 trace 完成率。
        "request_success_rate": 1.0 if attempted else 0.0,
        "request_success_definition": "error-free started requests; valid deadline partials accepted",
        "wall_time_sec": float(elapsed),
        "window_duration_sec": float(duration),
        "output_tokens": tokens,
        # 没有采集的延迟沿用既有-1缺失标记，不伪造为0或窗口时长。
        "ttft_ms_p50": _metric(raw, "median_ttft_ms", "p50_ttft_ms"),
        "ttft_ms_p99": _metric(raw, "p99_ttft_ms"),
        "tpot_ms_p50": _metric(raw, "median_tpot_ms", "p50_tpot_ms"),
        "tpot_ms_p99": _metric(raw, "p99_tpot_ms"),
        "output_throughput": throughput,
        "raw": _scalar_metrics(raw),
        "window": raw,
    }


def main() -> None:
    """旧协议先 gate；显式窗口协议按配置预热后测窗口、确认空闲，再执行 gate。"""
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: clean_benchmark.py <endpoint> <gate-data> <output-dir> "
            "<verifier-config.json>"
        )
    endpoint, data_path, output_dir_raw, config_raw = sys.argv[1:]
    config_path = Path(config_raw)
    config = json.loads(config_path.read_text())
    output_dir = Path(output_dir_raw)
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_kind = str(config["gate"]["kind"])
    window_enabled = isinstance(config.get("bench_serving"), dict) and "window_duration_sec" in config["bench_serving"]
    if window_enabled and (config.get("agentic_rollout") or gate_kind in (
        "video-reference", "action-reference", "realtime-video-reference", "audio-tts-reference",
    )):
        raise ValueError("固定窗口不能与 rollout 或 reference evaluator 混用")
    if gate_kind == "video-reference":
        import video_eval

        summary = video_eval.evaluate(endpoint, Path(data_path), output_dir, config_path)
        (output_dir / "gate-result.json").write_text(
            json.dumps(summary["gate"], indent=2) + "\n"
        )
        (output_dir / "benchmark-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return

    model_path = str(config["model_path"])
    served_model_name = str(config["served_model_name"])
    bench = load_bench_config(config, config_path)
    num_prompts = int(bench["num_prompts"])
    window_performance = None
    if window_enabled:
        if gate_kind not in {*gsm8k_eval.GSM8K_KINDS, *gpqa_eval.GPQA_KINDS, *longbench_eval.LONGBENCH_KINDS, "mmmu-openai-chat"}:
            raise ValueError(f"未知的 gate.kind：{gate_kind}")
        from fixed_window_benchmark import run_window, wait_for_idle

        # 窗口runner负责显式预热；窗口结束后排空，确保性能与真实gate不重叠。
        with (output_dir / "performance.log").open("w") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                raw_window = run_window(endpoint, output_dir / "window", config)
                print(json.dumps(raw_window, ensure_ascii=False), flush=True)
        (output_dir / "performance-result.jsonl").write_text(json.dumps(raw_window, ensure_ascii=False) + "\n")
        window_performance = _window_performance(raw_window, bench)
        wait_for_idle(
            endpoint,
            output_dir / "window-idle-before-gate.json",
            expected_engines=bench["window_expected_engines"],
            timeout_sec=180,
            backend=bench.get("window_idle_backend", bench["backend"]),
        )
    if gate_kind in gsm8k_eval.GSM8K_KINDS:
        gate = gsm8k_eval.evaluate(
            endpoint, Path(data_path), gsm8k_eval.load_eval_config(config_path)
        )
    elif gate_kind in gpqa_eval.GPQA_KINDS:
        gate = gpqa_eval.evaluate(
            endpoint, Path(data_path), gpqa_eval.load_eval_config(config_path)
        )
    elif gate_kind in longbench_eval.LONGBENCH_KINDS:
        gate = longbench_eval.evaluate(
            endpoint, Path(data_path), longbench_eval.load_eval_config(config_path)
        )
    elif gate_kind == "mmmu-openai-chat":
        gate = mmmu_eval.evaluate(
            endpoint,
            Path(data_path),
            {
                "served_model_name": served_model_name,
                # 可选的 smoke 口径题数;缺省 None = 全量(len(samples))。
                "num_questions": config["gate"].get("num_questions"),
            },
        )
    else:
        raise ValueError(f"未知的 gate.kind：{gate_kind}")
    (output_dir / "gate-result.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({"gate": gate}, ensure_ascii=False), flush=True)

    # agentic rollout 模式:正确性与性能由 verifier host 侧的真实 rollout 给出,
    # 本 Job 只负责可信 gate;写 gate-only summary 后直接返回。
    if config.get("agentic_rollout"):
        summary = {"gate": gate, "performance": None, "mode": "gate-only"}
        (output_dir / "benchmark-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return

    if window_performance is not None:
        summary = {"gate": gate, "performance": window_performance}
        (output_dir / "benchmark-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return

    bench_result_path = output_dir / "performance-result.jsonl"
    bench_log_path = output_dir / "performance.log"
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_official_bench.py")),
        "--backend",
        str(bench["backend"]),
        "--base-url",
        endpoint,
        "--model",
        model_path,
        # sglang 原生 backend 的请求不带 model 字段,而新版 bench 的
        # model_id = served_model_name or model 且 image/mmmu 数据集用
        # model_id 加载 processor——传 served name 会让无外网的 pod 去连
        # HF。仅 OpenAI 系 backend(请求需要 model 字段)才传 served name。
        *(
            []
            if str(bench["backend"]) == "sglang"
            else ["--served-model-name", served_model_name]
        ),
        "--tokenizer",
        model_path,
        "--dataset-name",
        str(bench["dataset_name"]),
        "--random-input-len",
        str(bench["random_input_len"]),
        "--random-output-len",
        str(bench["random_output_len"]),
        "--random-range-ratio",
        str(bench["random_range_ratio"]),
        "--max-concurrency",
        str(bench["max_concurrency"]),
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        "inf",
        "--seed",
        str(bench["seed"]),
        "--output-file",
        str(bench_result_path),
        "--output-details",
        *(["--mooncake-arrival-mode", bench["arrival_mode"]] if "arrival_mode" in bench else []),
        # 数据集特有参数(如 image 数据集的 --image-count/--image-resolution)
        # 原样追加;无需要时配置为 []。
        *[str(arg) for arg in bench["extra_args"]],
    ]
    process_env = {
        **os.environ,
        "NO_PROXY": "*",
        "no_proxy": "*",
        "OMP_NUM_THREADS": "1",
        "HARBOR_BENCH_SERVING": str(bench["script_path"]),
    }
    started_at = time.perf_counter()
    with bench_log_path.open("w") as log_file:
        completed = subprocess.run(
            command,
            env=process_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            # 大模型(如 744B)在 agentic 回放下 90min 跑不完;按 task 配置放宽
            timeout=int(bench.get("bench_timeout_sec", 5400)),
            check=False,
        )
    wall_time_sec = time.perf_counter() - started_at
    if completed.returncode != 0:
        raise RuntimeError(
            f"bench_serving 失败，退出码 {completed.returncode}：\n"
            f"{bench_log_path.read_text()}"
        )

    raw = _last_json_object(bench_result_path)
    arrival_metadata = {}
    if bench["dataset_name"] == "mooncake":
        mode = bench.get("arrival_mode", "trace")
        protocol = f"mooncake-full-replay-{mode}-v1"
        if raw.get("arrival_mode", "trace") != mode or raw.get("measurement_protocol", "mooncake-full-replay-trace-v1") != protocol:
            raise ValueError("Mooncake arrival/protocol与配置不一致")
        if mode == "asap" and raw.get("request_rate") != "inf":
            raise ValueError("ASAP回放缺少无到达节流的request_rate证据")
        arrival_metadata = {"arrival_mode": mode, "measurement_protocol": protocol,
                            "request_rate": "inf" if mode == "asap" else "trace"}
    completed_requests = int(_metric(raw, "completed", "successful_requests"))
    requested = _requested_count(bench, num_prompts)
    summary = {
        "gate": gate,
        "performance": {
            **arrival_metadata,
            "num_prompts": num_prompts,
            "requests_attempted": requested,
            "max_concurrency": int(bench["max_concurrency"]),
            "random_input_len": int(bench["random_input_len"]),
            "random_output_len": int(bench["random_output_len"]),
            "seed": int(bench["seed"]),
            "completed": completed_requests,
            "request_success_rate": completed_requests / requested,
            "wall_time_sec": round(wall_time_sec, 3),
            "ttft_ms_p50": _metric(raw, "median_ttft_ms", "p50_ttft_ms"),
            "ttft_ms_p99": _metric(raw, "p99_ttft_ms"),
            "tpot_ms_p50": _metric(raw, "median_tpot_ms", "p50_tpot_ms"),
            "tpot_ms_p99": _metric(raw, "p99_tpot_ms"),
            "output_throughput": _metric(raw, "output_throughput"),
            "raw": _scalar_metrics(raw),
        },
    }
    (output_dir / "benchmark-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
