#!/usr/bin/env python3
"""在真实 workload 的稳定期内对 SGLang 服务做**有界窗口** torch profile,并把
chrome trace 直接聚合成「热点 kernel + GPU 空转 + host 算子溯源」报告。

它解决裸用 `/start_profile` 的三个坑:
1. 全程 profile → trace 几个 GB、含 warmup/CUDA graph 捕获,热点被污染;
   这里 workload 先跑满 warmup 再开窗,只采 `--steps` 个 forward step。
2. `.trace.json.gz` 要手动翻 Chrome,看不出 GPU 到底闲多久;
   这里直接算 GPU busy/idle 与最大 bubble。
3. kernel 名字回不到 PyTorch 层;这里用 correlation id +
   cuda_runtime/cpu_op 嵌套关系把每个热点 kernel 溯源到发起它的 host 算子。

参数:
  out_dir          输出目录(trace、workload 日志、report.json 都落这里)
  --steps N        采样窗口的 forward step 数(默认 20;越大越稳,trace 越大)
  --warmup-sec S   开窗前先让 workload 跑多久(默认 90 秒,躲开首轮编译/捕获)
  --               之后全部参数是要执行的 workload 命令(必需)

服务地址取自 `HARBOR_SERVICE_PORT`(缺失直接报错),固定用 `http://[::1]:$PORT`。

产物 `<out_dir>/`:`report.json`(采样时服务端真实生效的 `server_args`、窗口
`gpu_busy_pct` / `gpu_idle_us`、最大设备 bubble `top_gaps`、按 `pct_of_gpu_busy`
排序的热点 kernel 表,每个 kernel 附 `host_ops`)、`traces/*.trace.json.gz`、
`workload.log`。

读法与纪律(踩过的坑):
- **先看 `gpu_busy_pct`,再看 kernel 表**。>90% 说明是算力/kernel 瓶颈,去抠
  `hotspots` 头部;busy 低而 `top_gaps` 有毫秒级 bubble 说明瓶颈在 host 侧
  launch/调度/同步,这时在 kernel 表里抠百分之几是白费力气。
- `host_ops` 是改动入口:它直接告诉你热点 kernel 由哪个 Python 调用点发起。
  **但热点行的 `host_ops` 读出 `<cudaGraphLaunch>` 时是正常现象,不是工具坏了**:
  CUDA graph replay 的 kernel 没有外层 `aten::` 作用域,correlation id 也被一次
  replay 里的上百个 kernel 共用,所以溯源在 graph 边界处天然断掉
  (2026-09-10 真机实测:graph replay 占 device 事件 90.1%、`gpu_busy` 98.91%;
  eager 路径的溯源成功率 95%,是好的)。要拿到热点的真实 host 溯源,**用
  `--disable-cuda-graph` 另起一轮服务专门做溯源**(这一轮的吞吐更加不能当性能数
  据用);粗粒度的替代读数是 trace 里的 `step[...]` user_annotation。
- `memcpy_memset_us` 偏高通常意味着多余的 H2D 拷贝或 `.cpu()`/`.item()` 同步。
- **窗口要落在稳定期**。首轮请求含权重加载后的 lazy init、CUDA graph 捕获、
  autotune,采进去会把热点带偏;长上下文 replay 建议 `--warmup-sec` ≥120。
- **在评分用的 workload 上 profile**。单一长度压测的 prefill/decode 配比和
  prefix cache 命中都不是真实形态,在它上面调优会过拟合。
- **开窗那一轮的吞吐不是性能数据**(profiler 有开销)。优化前后的正式对比必须
  另跑一轮不带 profile 的完整 workload,同 workload、同参数、同 seed。
- `--steps` 的单位是**调度器 batch step**,不是模型 forward。二者不是 1:1 ——
  2026-09-10 llada 真机实测 `--steps 60` 展开成 768 次模型 forward(≈12.8×),
  采出 10.29s / 50 万条 device 事件的窗口。所以别开太大:20~60 步足够定档,
  上百步只是让 trace 变大、解析变慢。
- 这是调试工具,不要把它加进交付层,也不要在 `launch.sh` 里调用它。

示例(在 agentic replay workload 的稳定期采 20 步):
  python3 /mnt/rsibench/rsibench/profile_sglang.py \\
    /tmp/prof-baseline --steps 20 --warmup-sec 120 -- \\
    python3 /mnt/rsibench/rsibench/vendored_bench_serving.py \\
      --backend sglang --base-url "http://[::1]:${HARBOR_SERVICE_PORT}" \\
      --model /mnt/rsibench/qwen3p8_27b_fp8 \\
      --tokenizer /mnt/rsibench/qwen3p8_27b_fp8 \\
      --dataset-name mooncake \\
      --dataset-path /mnt/rsibench/rsibench/agentic-trace-64k.jsonl \\
      --mooncake-slowdown-factor 1.0 --max-concurrency 4 \\
      --num-prompts 10000 --request-rate inf --seed 42
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# torch profiler chrome trace 里的设备侧类别
KERNEL_CATS = {"kernel"}
MEMORY_CATS = {"gpu_memcpy", "gpu_memset"}
RUNTIME_CATS = {"cuda_runtime", "cuda_driver"}
TOP_N = 25
# 等服务端采满 num_steps 并自动停止的上限;超时就强制 /stop_profile。
CAPTURE_TIMEOUT_SEC = 600.0
# 停止后等 trace 写盘/大小稳定的上限,以及判定「稳定」的采样间隔。
STOP_FLUSH_SEC = 300.0
STABLE_GAP_SEC = 3.0

# 目标恒为本机 [::1],必须绕开环境里的 http_proxy(否则代理回 403)。
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def base_url() -> str:
    """服务地址:pod 内 localhost 只解析 IPv4,服务绑 IPv6,固定走 [::1]。"""
    port = os.environ["HARBOR_SERVICE_PORT"]
    return f"http://[::1]:{port}"


def http_json(url: str, payload: dict | None = None, timeout: float = 3600.0) -> dict:
    """GET/POST JSON;非 2xx 由 urllib 直接抛,不做兜底。"""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def http_text(url: str, payload: dict | None = None, timeout: float = 120.0) -> str:
    """POST 并按纯文本读取——`/start_profile`、`/stop_profile` 返回的是
    `Start profiling.` 这类裸文本,不是 JSON。"""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with OPENER.open(req, timeout=timeout) as resp:
        return resp.read().decode().strip()


def wait_for_traces(
    trace_dir: Path, existing: set, proc: subprocess.Popen, url: str
) -> list[Path]:
    """等异步 profiler 落盘的文件出现。

    `/start_profile` 立即返回,采满 `num_steps` 后由服务端自行停止并写盘,
    所以要轮询;新 trace 迟迟不出现就强制 `/stop_profile`。文件出现 ≠ 写完,
    完整性由 `load_trace` 的重试保证。
    """
    deadline = time.time() + CAPTURE_TIMEOUT_SEC
    forced = False
    while True:
        traces = sorted(set(trace_dir.glob("*.trace.json.gz")) - existing)
        if traces:
            print(f"[profile] trace 已出现:{[p.name for p in traces]}")
            return traces
        if proc.poll() is not None:
            raise RuntimeError(
                f"workload 在采集窗口内就退出了(rc={proc.returncode}),"
                f"没跑满设定的 forward step"
            )
        if time.time() > deadline:
            if forced:
                raise RuntimeError(f"{trace_dir} 下始终没有新 trace,采集失败")
            print("[profile] 未自动停止,强制 /stop_profile")
            print(f"[profile] stop_profile -> {http_text(f'{url}/stop_profile', {})}")
            forced = True
            deadline = time.time() + STOP_FLUSH_SEC
        time.sleep(2.0)


def server_provenance(url: str) -> dict:
    """快照服务端真实生效的配置——profile 结论必须能对上是哪份配置。"""
    info = http_json(f"{url}/get_server_info")
    args = info.get("server_args", info)
    keys = (
        "model_path",
        "tp_size",
        "context_length",
        "mem_fraction_static",
        "max_running_requests",
        "chunked_prefill_size",
        "cuda_graph_max_bs",
        "speculative_algorithm",
        "speculative_num_steps",
        "attention_backend",
        "mamba_full_memory_ratio",
        # 下面几项服务于共用本脚本的另外两道题(dLLM / 多模态),缺失就跳过
        "dllm_algorithm",
        "enable_multimodal",
        "page_size",
        "disable_radix_cache",
        "quantization",
    )
    return {key: args[key] for key in keys if key in args}


def load_trace(path: Path) -> list[dict]:
    """读 chrome trace(.json.gz),只留有时长的完整事件 X。

    服务端是边采边刷地写这个文件,「写完了」的唯一可靠判据就是能整段解出来:
    半截文件在 gzip 层抛 EOFError。所以在 STOP_FLUSH_SEC 内轮询重试,超时直接抛。
    """
    deadline = time.time() + STOP_FLUSH_SEC
    while True:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                doc = json.load(stream)
            break
        except (EOFError, gzip.BadGzipFile, json.JSONDecodeError):
            if time.time() > deadline:
                raise
            time.sleep(STABLE_GAP_SEC)
    return [
        ev
        for ev in doc["traceEvents"]
        if ev.get("ph") == "X" and ev.get("dur") is not None
    ]


def host_op_by_correlation(events: list[dict]) -> dict[int, str]:
    """correlation id → 发起该 kernel 的最内层 host 算子名。

    做法:同一 tid 上 cpu_op 与 cuda_runtime 按 (ts, -dur) 归并扫描,用栈维护
    嵌套关系;runtime 事件出现时栈顶就是它的最内层 cpu_op。一遍 O(n log n)。
    """
    per_tid: dict[tuple, list[dict]] = {}
    for ev in events:
        cat = ev.get("cat")
        if cat == "cpu_op" or cat in RUNTIME_CATS:
            per_tid.setdefault((ev.get("pid"), ev.get("tid")), []).append(ev)

    mapping: dict[int, str] = {}
    for group in per_tid.values():
        group.sort(key=lambda ev: (ev["ts"], -ev["dur"]))
        stack: list[tuple[float, str]] = []  # (end_ts, name)
        for ev in group:
            while stack and stack[-1][0] <= ev["ts"]:
                stack.pop()
            if ev.get("cat") == "cpu_op":
                stack.append((ev["ts"] + ev["dur"], ev["name"]))
                continue
            corr = ev.get("args", {}).get("correlation")
            if corr is not None:
                # 栈空说明这次 launch 不在任何 aten:: 作用域内,最常见的原因是
                # **CUDA graph replay**:引擎直接调 graph.replay(),cudaGraphLaunch
                # 全部落在 aten 之外,且一个 correlation id 会扇出上百个 device
                # 事件(2026-09-10 llada 真机实测:768 次 graph launch 每个扇出
                # 592 个 kernel,占全部 device 事件的 90.1%、gpu_busy 的 98.91%)。
                # 此时**不能留空**——空的 host_ops 与"工具坏了"无法区分,而
                # host_ops 正是本脚本文档里指的"改动入口"。退回记 launch API
                # 本身,读出来是 <cudaGraphLaunch>,含义明确。
                mapping[corr] = stack[-1][1] if stack else f"<{ev['name']}>"
    return mapping


def merge_busy(spans: list[tuple[float, float]]) -> float:
    """区间并集总长度:多流重叠时不能简单求和,否则 busy 会超过墙钟。"""
    total = 0.0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    return total + cur_end - cur_start


def top_gaps(spans: list[tuple[float, float]], limit: int = 5) -> list[dict]:
    """设备侧最大空转 bubble——decode 阶段 launch-bound 会直接体现在这里。"""
    gaps = []
    prev_end = spans[0][1]
    for start, end in spans[1:]:
        if start > prev_end:
            gaps.append({"start_us": prev_end, "gap_us": round(start - prev_end, 1)})
        prev_end = max(prev_end, end)
    gaps.sort(key=lambda item: item["gap_us"], reverse=True)
    return gaps[:limit]


def analyze(path: Path) -> dict:
    """把单个 rank 的 trace 聚合成热点表 + GPU 占用 + host 溯源。"""
    events = load_trace(path)
    device = [ev for ev in events if ev.get("cat") in KERNEL_CATS | MEMORY_CATS]
    if not device:
        raise RuntimeError(f"{path} 里没有设备侧事件,窗口没落在真实前向上")

    corr_to_host = host_op_by_correlation(events)
    spans = sorted((ev["ts"], ev["ts"] + ev["dur"]) for ev in device)
    busy_us = merge_busy(spans)
    span_us = spans[-1][1] - spans[0][0]

    agg: dict[str, dict] = {}
    for ev in device:
        entry = agg.setdefault(
            ev["name"],
            {"count": 0, "total_us": 0.0, "cat": ev["cat"], "host_ops": {}},
        )
        entry["count"] += 1
        entry["total_us"] += ev["dur"]
        host = corr_to_host.get(ev.get("args", {}).get("correlation"))
        if host is not None:
            entry["host_ops"][host] = entry["host_ops"].get(host, 0) + 1

    ranked = sorted(agg.items(), key=lambda kv: kv[1]["total_us"], reverse=True)
    hotspots = []
    for name, entry in ranked[:TOP_N]:
        hosts = sorted(entry["host_ops"].items(), key=lambda kv: kv[1], reverse=True)
        hotspots.append(
            {
                "kernel": name,
                "cat": entry["cat"],
                "count": entry["count"],
                "total_us": round(entry["total_us"], 1),
                "mean_us": round(entry["total_us"] / entry["count"], 2),
                "pct_of_gpu_busy": round(100 * entry["total_us"] / busy_us, 2),
                "host_ops": [name for name, _ in hosts[:3]],
            }
        )

    memory_us = sum(ev["dur"] for ev in device if ev["cat"] in MEMORY_CATS)
    return {
        "trace_file": path.name,
        "window": {
            "span_us": round(span_us, 1),
            "gpu_busy_us": round(busy_us, 1),
            "gpu_idle_us": round(span_us - busy_us, 1),
            "gpu_busy_pct": round(100 * busy_us / span_us, 2),
            "device_event_count": len(device),
            "distinct_kernels": len(agg),
            "memcpy_memset_us": round(memory_us, 1),
        },
        "top_gaps": top_gaps(spans),
        "hotspots": hotspots,
    }


def run_workload(command: list[str], log_path: Path) -> subprocess.Popen:
    """独立进程组启动 workload,便于窗口结束后整组收掉。"""
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )


def stop_workload(proc: subprocess.Popen) -> None:
    """先 SIGTERM 整组,给 10 秒收尾,不退就 SIGKILL。"""
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    deadline = time.time() + 10
    while time.time() < deadline and proc.poll() is None:
        time.sleep(0.5)
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def render(report: dict) -> str:
    """给人看的 markdown 摘要(report.json 里是完整数据)。"""
    lines = []
    for rank in report["ranks"]:
        win = rank["window"]
        lines.append(f"### {rank['trace_file']}")
        lines.append(
            f"window {win['span_us'] / 1000:.1f} ms · GPU busy "
            f"{win['gpu_busy_pct']}% · idle {win['gpu_idle_us'] / 1000:.1f} ms · "
            f"{win['device_event_count']} events / {win['distinct_kernels']} kernels"
        )
        biggest = rank["top_gaps"]
        if biggest:
            lines.append(f"max device bubble: {biggest[0]['gap_us'] / 1000:.2f} ms")
        lines.append("")
        lines.append("| % busy | total ms | count | mean us | kernel | host op |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for item in rank["hotspots"]:
            host = ", ".join(item["host_ops"]) or "-"
            lines.append(
                f"| {item['pct_of_gpu_busy']} | {item['total_us'] / 1000:.2f} | "
                f"{item['count']} | {item['mean_us']} | `{item['kernel'][:70]}` | "
                f"{host} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        usage="%(prog)s out_dir [--steps N] [--warmup-sec S] -- <workload cmd ...>",
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-sec", type=float, default=90.0)

    argv = sys.argv[1:]
    if "--" not in argv:
        parser.error("必须用 `--` 分隔本脚本参数与 workload 命令")
    split = argv.index("--")
    args = parser.parse_args(argv[:split])
    command = argv[split + 1 :]
    if not command:
        parser.error("`--` 之后必须给出 workload 命令")

    url = base_url()
    out_dir = args.out_dir
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    existing = set(trace_dir.glob("*.trace.json.gz"))

    info = server_provenance(url)
    print(f"[profile] server @ {url}: {json.dumps(info, ensure_ascii=False)}")

    proc = run_workload(command, out_dir / "workload.log")
    try:
        print(f"[profile] workload pid={proc.pid}, warmup {args.warmup_sec}s")
        time.sleep(args.warmup_sec)
        if proc.poll() is not None:
            raise RuntimeError(
                f"workload 在 warmup 内就退出了(rc={proc.returncode}),"
                f"看 {out_dir / 'workload.log'}"
            )

        print(f"[profile] capturing {args.steps} forward steps ...")
        started = time.time()
        reply = http_text(
            f"{url}/start_profile",
            {
                "output_dir": str(trace_dir),
                "num_steps": args.steps,
                "activities": ["CPU", "GPU"],
                "with_stack": False,
                "record_shapes": True,
                "profile_prefix": "harbor-window",
            },
        )
        print(f"[profile] start_profile -> {reply!r}")
        traces = wait_for_traces(trace_dir, existing, proc, url)
        print(f"[profile] done in {time.time() - started:.1f}s")

    finally:
        stop_workload(proc)

    report = {
        "base_url": url,
        "server_args": info,
        "workload": command,
        "capture": {
            "num_steps": args.steps,
            "warmup_sec": args.warmup_sec,
            "activities": ["CPU", "GPU"],
        },
        "ranks": [analyze(path) for path in traces],
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(render(report))
    print(
        "[profile] 注意:开窗期间 torch profiler 本身有开销,这一轮的 "
        "output throughput 不能当性能数据用;性能对比请另跑一轮不带 profile 的 "
        "完整 workload。"
    )
    print(f"[profile] report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    sys.exit(main())
