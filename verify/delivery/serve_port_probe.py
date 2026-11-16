"""断言服务端口的实际应答方是本 trial 启动的服务进程树。

参数：port 为服务合约端口；pid_file 为 serve_controller 落盘的 rank-0.pid；
--endpoint / --log 提供 nonce 仲裁所需的服务地址与本服务的 stdout 日志。

为什么需要它：Merlin GPU pod 用宿主机网络，同一台机器上任何 job 都能抢先
绑走服务端口——launch.sh 合约端口 8000 同时也是全公司 sglang/vLLM 的默认
端口。health 探针只看「有服务返回 200」，分不出 200 来自谁。0902 实录
（historical validation infer-llada2p1mini-16b）：本 trial 的 serve 实例绑 :8000
EADDRINUSE 秒退，同宿主机另一 job 的 Qwen3 thinking 服务把 gate/benchmark
全程接走，gate accuracy 0.758 / invalid 0.202 全是别人模型的成绩，
verifier 却报 deploy_success=1。

判定分两层，任一通过即 OWNED：

1. **进程树 × socket 表**：launch 进程树（rank-0.pid 及后代）持有的 socket
   inode 与 /proc/net/tcp{,6} 中 LISTEN :port 行求交。注意共享宿主机 netns
   时该表是**全机**socket 表且为 seq_file 遍历，繁忙宿主机上单次读取可能
   漏行（0902 深夜首版探针据此误杀了健康部署），所以带多轮重试，且不匹配
   不直接定罪。
2. **nonce 端到端仲裁**：向 endpoint 发一条唯一路径请求（404 无妨，nonce 同时
   放在路径和 X-Request-ID 头），再在本服务的 stdout 日志里找该 nonce——
   uvicorn/sglang 逐条记录 access log 路径，Ray Serve 记录请求 ID。
   谁应答端口，nonce 就落在谁的日志里，与 netns/pidns/表遍历完全无关。

两层都不过才 FOREIGN（exit 3），并打印双侧诊断（表里的 :port 行、进程树
规模与其 LISTEN 端口）供归因。约束：launch.sh 及服务进程须保持前台进程树
（现有 task 都是 `exec` 常驻）；若引擎既 daemonize 又不记 access log，会
被误判——届时诊断输出会直接指出。

退出码：0=OWNED；2=端口无监听且仲裁失败；3=监听者/应答方不是本服务；
4=pid 文件缺失。

示例：
    sudo python3 serve_port_probe.py 8000 /workspace/deploy/rank-0.pid \
        --endpoint 'http://[2605:340::1]:8000' --log /workspace/deploy/rank-0.log
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# 环境变量可调,供单测把重试压到毫秒级;生产默认多轮重试对抗表遍历漏行。
TABLE_ATTEMPTS = int(os.environ.get("PORT_PROBE_TABLE_ATTEMPTS", "4"))
TABLE_RETRY_SEC = float(os.environ.get("PORT_PROBE_TABLE_RETRY_SEC", "2"))
NONCE_WAIT_SEC = float(os.environ.get("PORT_PROBE_NONCE_WAIT_SEC", "2"))


def listen_rows(port: int | None = None) -> list[tuple[int, str]]:
    """读取 /proc/net/tcp{,6} 的 LISTEN 行，返回 (端口, inode)。

    共享宿主机 netns 时这是全机表；port=None 返回全部 LISTEN 行。
    """
    rows: list[tuple[int, str]] = []
    for name in ("tcp", "tcp6"):
        path = Path("/proc/net") / name
        if not path.is_file():
            continue
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10:
                continue
            local, state, inode = fields[1], fields[3], fields[9]
            # 0A = TCP_LISTEN；本地地址形如 "…:1F40"（端口为十六进制）。
            if state != "0A":
                continue
            row_port = int(local.rsplit(":", 1)[1], 16)
            if port is None or row_port == port:
                rows.append((row_port, inode))
    return rows


def _visible_pids() -> list[int]:
    return sorted(
        int(entry.name) for entry in Path("/proc").iterdir() if entry.name.isdigit()
    )


def descendant_pids(root: int) -> set[int]:
    """返回 root 及其全部后代（launch.sh 的 setsid 进程组按 ppid 链可达）。"""
    children: dict[int, list[int]] = {}
    for pid in _visible_pids():
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            continue
        # comm 字段可含空格/括号，ppid 是最后一个右括号后的第 2 个字段。
        ppid = int(stat.rpartition(")")[2].split()[1])
        children.setdefault(ppid, []).append(pid)
    pids = {root}
    queue = [root]
    while queue:
        for child in children.get(queue.pop(), []):
            if child not in pids:
                pids.add(child)
                queue.append(child)
    return pids


def tree_socket_inodes(pids: set[int]) -> set[str]:
    """收集进程集持有的全部 socket inode（fd 枚举可靠，不受表遍历影响）。"""
    inodes: set[str] = set()
    for pid in pids:
        fd_dir = Path(f"/proc/{pid}/fd")
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(fd)
            except OSError:
                continue
            if link.startswith("socket:["):
                inodes.add(link[8:-1])
    return inodes


def nonce_hits_log(endpoint: str, log_path: Path) -> tuple[bool, str]:
    """向 endpoint 发唯一路径请求，检查本服务日志是否记录到该 nonce。"""
    nonce = f"harbor-port-probe-{uuid.uuid4().hex}"
    # nonce 同时放进路径与 X-Request-ID：uvicorn/sglang 的 access log 记原始路径,而 Ray Serve
    # 这类网关只记匹配到的路由(0910 verl-tinker 实录:探针请求被记成 `GET / 404`)但会原样
    # 记录请求 ID。两种载体都指向同一 nonce,判定语义不变:谁应答,nonce 就落在谁的日志里。
    request = urllib.request.Request(f"{endpoint}/{nonce}", headers={"X-Request-ID": nonce})
    # 直连服务地址，绕开代理环境变量。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=15):
            pass
    except urllib.error.HTTPError:
        pass  # 404/405 都行,只要有进程应答并留痕。
    except OSError as exc:
        return False, f"nonce 请求失败:{exc}"
    time.sleep(NONCE_WAIT_SEC)
    try:
        logged = nonce in log_path.read_text(errors="replace")
    except OSError as exc:
        return False, f"读服务日志失败:{exc}"
    return logged, f"nonce={nonce}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument("pid_file", type=Path)
    parser.add_argument("--endpoint", default="", help="服务地址,启用 nonce 仲裁")
    parser.add_argument("--log", type=Path, default=None, help="本服务 stdout 日志")
    args = parser.parse_args()

    if not args.pid_file.is_file():
        print(f"NOPIDFILE {args.pid_file}（serve_controller 未启动过服务）")
        return 4
    root = int(args.pid_file.read_text().split()[0])

    port_rows: list[tuple[int, str]] = []
    ours: set[str] = set()
    tree: set[int] = set()
    for attempt in range(1, TABLE_ATTEMPTS + 1):
        tree = descendant_pids(root)
        ours = tree_socket_inodes(tree)
        port_rows = listen_rows(args.port)
        if any(inode in ours for _, inode in port_rows):
            print(f"OWNED port={args.port} via=socket-table attempt={attempt}")
            return 0
        time.sleep(TABLE_RETRY_SEC)

    nonce_detail = "未启用(缺 --endpoint/--log)"
    if args.endpoint and args.log is not None:
        hit, nonce_detail = nonce_hits_log(args.endpoint, args.log)
        if hit:
            print(f"OWNED port={args.port} via=nonce（socket 表未命中,详见诊断）")
            _print_diag(args.port, port_rows, tree, ours)
            return 0

    _print_diag(args.port, port_rows, tree, ours)
    if not port_rows:
        print(f"NOLISTENER port={args.port} 且 nonce 仲裁未通过({nonce_detail})")
        return 2
    print(
        f"FOREIGN port={args.port} nonce仲裁={nonce_detail}"
        "（监听者/应答方不属于本服务进程树,疑同宿主机其它 job）"
    )
    return 3


def _print_diag(port: int, port_rows: list, tree: set[int], ours: set[str]) -> None:
    """双侧诊断:表里的 :port 行 + 我方进程树的 LISTEN 端口。"""
    our_listen_ports = sorted(
        {p for p, inode in listen_rows() if inode in ours}
    )
    print(
        f"DIAG rows(:{port})={port_rows[:5]} tree_pids={len(tree)} "
        f"tree_sockets={len(ours)} tree_listen_ports={our_listen_ports[:10]}"
    )


if __name__ == "__main__":
    sys.exit(main())
