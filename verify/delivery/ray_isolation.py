"""校验逐 pod 启动的 Ray 端口与节点归属，不启动或停止 Ray。

参数：操作 allocate/member/cluster，以及 API 端口或显式 GCS IPv4:port。
allocate 只从平台 PORT<n> 中按冻结的 Ray 2.56 CLI 契约选择；member 检查
controller rc/PID 与本 pod 的进程/socket；cluster 只查询。
示例：PORT0=10844 PORT1=10279 PORT5=9418 python3 ray_isolation.py allocate 10844
"""

from __future__ import annotations

import errno
import ipaddress
import json
import os
import shlex
import socket
import sys
from pathlib import Path

# 明确冻结的通用端口契约，来源为实机 Ray 2.56.0 的 ray/scripts/scripts.py：
# CLI min/max-worker-port 默认 10002/19999（第 515–528 行），parameter.py
# 第 325–348 行将整个闭区间列入组件预留集合。不是系统端口占用的推测。
RAY_PORT_CONTRACT = "ray-2.56.0-cli-defaults"
RAY_WORKER_PORT_RANGE = (10002, 19999)
# 同版本启动异常确认的 client/dashboard/dashboard-agent-http 默认固定端口。
RAY_FIXED_PORTS = frozenset({10001, 8265, 52365})
DEPLOY_DIR = Path("/workspace/deploy")


def compatible_ray_port(port: int, api_port: int) -> bool:
    """判断端口是否符合冻结契约；不查询系统、不选择任何未分配端口。"""
    low, high = RAY_WORKER_PORT_RANGE
    return (type(port) is int and 1024 <= port <= 65535 and port != api_port
            and not low <= port <= high and port not in RAY_FIXED_PORTS)


def _assert_port_available(port: int) -> None:
    """短暂 bind 两个地址族，绝不设置复用、保留监听或停止已有监听者。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ipv4:
        ipv4.bind(("0.0.0.0", port))
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as ipv6:
            ipv6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            ipv6.bind(("::", port))


def allocated_port(api_port: int) -> dict:
    """只检查 head 实际 PORT<n>，排除 API/保留范围/占用；无候选则立即失败。"""
    allocations = []
    for name, value in os.environ.items():
        index = name.removeprefix("PORT")
        if not name.startswith("PORT") or not index.isascii() or not index.isdecimal():
            continue
        if not value.isascii() or not value.isdecimal() or not 1024 <= int(value) <= 65535:
            raise RuntimeError(f"Invalid platform allocation {name}={value!r}")
        allocations.append((int(index), name, int(value)))
    if not allocations:
        raise RuntimeError("Ray isolation requires platform head PORT<n> allocations")
    for _, name, port in sorted(allocations):
        if not compatible_ray_port(port, api_port):
            continue
        try:
            _assert_port_available(port)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            continue  # 仅尝试列表中的下一个已分配端口，不搜索系统端口空间。
        return {"port": port, "source": f"head:{name}", "contract": RAY_PORT_CONTRACT}
    raise RuntimeError(
        f"No available platform-assigned Ray port outside API/reserved ports: "
        f"contract={RAY_PORT_CONTRACT}, worker_range={RAY_WORKER_PORT_RANGE}"
    )


def validate_ray_address(address: str) -> str:
    """仅接受显式 IPv4:合法端口；在导入 Ray 前拒绝 local/auto/URI 等输入。"""
    host, separator, port = address.rpartition(":")
    if not separator or not port.isascii() or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError("Ray isolation requires an explicit IPv4:port address")
    try:
        ipv4 = ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError as exc:
        raise ValueError("Ray isolation requires an explicit IPv4:port address") from exc
    if ipv4.is_unspecified or ipv4.is_multicast:
        raise ValueError("Ray isolation requires a concrete IPv4 peer address")
    return f"{ipv4}:{int(port)}"


def _flag(argv: list[str], name: str) -> str:
    """读取 Ray 进程的显式参数，兼容连字符/下划线拼写而不猜测值。"""
    for index, arg in enumerate(argv):
        key, separator, value = arg.partition("=")
        if key.lstrip("-").replace("-", "_") == name:
            return value if separator else argv[index + 1]
    raise RuntimeError(f"Ray process missing explicit {name}")


def _owns_listener(pid: str | None, port: int) -> bool | None:
    """返回 True=全属本进程、False=存在非 owned 监听、None=尚无监听。"""
    inodes = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(f"socket:[{fields[9]}]")
    if not inodes:
        return None
    if pid is None:
        return False
    owned = set()
    try:
        for fd in (Path("/proc") / pid / "fd").iterdir():
            try:
                owned.add(os.readlink(fd))
            except FileNotFoundError:  # 进程正常关闭 fd 的短暂竞态。
                continue
    except (FileNotFoundError, ProcessLookupError):
        # 进程消失后重新看监听：只有确实无 socket 才属于 startup-not-ready。
        return _owns_listener(None, port)
    if inodes <= owned:
        return True
    # socket 表与 fd 快照之间可能正常关闭监听；仍有监听但不能归属则失败。
    return _owns_listener(None, port)


def _ray_processes():
    """枚举本 pod 的 Ray 进程；只忽略进程消失，权限错误必须向上传播。"""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            argv = proc.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if argv and argv[0] and Path(argv[0]).name in {"raylet", "gcs_server"}:
            yield proc.name, argv


LAUNCH_CONFIG_PATH = Path("/logs/verifier/verifier-config.json")


def _controller_launch_running(rank: int) -> bool:
    """交叉核对既有 rc/PID、可信源码目录与独立进程组；False 只表示明确 rc=0。"""
    rc_path = DEPLOY_DIR / f"rank-{rank}.rc"

    def exit_code():
        """非零 rc 无条件失败；零 rc 只供后续 Ray 正面证据裁决。"""
        if rc_path.exists():
            rc = int(rc_path.read_text().strip())
            if rc != 0:
                raise RuntimeError(f"Controller launch exited: rank={rank}, rc={rc}")
            return rc
        return None

    exit_code()
    pid_path = DEPLOY_DIR / f"rank-{rank}.pid"
    if not pid_path.exists():
        raise RuntimeError(f"Controller launch PID missing: rank={rank}")
    pid = int(pid_path.read_text().strip())
    if pid <= 0:
        raise RuntimeError(f"Invalid controller launch PID: rank={rank}, pid={pid}")
    repo = Path(json.loads(LAUNCH_CONFIG_PATH.read_text())["engine_repo"])
    launch = repo / "launch.sh"
    if not repo.is_absolute() or not launch.is_file():
        raise RuntimeError("Controller launch source missing or invalid")
    # 对应 serve_controller.launch_wrapper_command 与 Popen(start_new_session=True)。
    expected = f'bash {shlex.quote(str(launch))}; echo "$?" > {shlex.quote(str(rc_path))}'
    proc = Path("/proc") / str(pid)
    try:
        os.kill(pid, 0)  # 只查存活/权限，不发送终止信号。
        state = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[0]
        if state in {"Z", "X"}:
            if exit_code() == 0:
                return False
            raise RuntimeError(f"Controller launch exited: rank={rank}, pid={pid}, state={state}")
        argv = proc.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
        if (len(argv) != 3 or Path(argv[0]).name != "bash" or argv[1:] != ["-c", expected]
                or Path(os.readlink(proc / "cwd")).resolve() != repo.resolve()
                or os.getpgid(pid) != pid or os.getsid(pid) != pid):
            raise RuntimeError(f"Foreign controller launch PID: rank={rank}, pid={pid}")
    except (ProcessLookupError, FileNotFoundError) as exc:
        if exit_code() == 0:
            return False
        raise RuntimeError(f"Controller launch exited: rank={rank}, pid={pid}") from exc
    return exit_code() is None


def local_member(address: str, rank: int) -> dict:
    """活 launch 可有界等待；明确 rc=0 的启动器必须立即有本 pod Ray 正面证据。"""
    address = validate_ray_address(address)
    running = _controller_launch_running(rank)

    def pending(reason):
        """已正常退出却缺 Ray 证据不是 startup-not-ready，不能空等到截止时间。"""
        if not running:
            raise RuntimeError(f"Controller launch exited with rc=0 without verified Ray node: rank={rank}, {reason}")
        return {"ready": False, "reason": reason}

    raylets, gcs = [], []
    for pid, argv in _ray_processes():
        if Path(argv[0]).name == "raylet":
            if _flag(argv, "gcs_address") != address:
                raise RuntimeError("Foreign Raylet GCS address in target pod")
            raylets.append(argv)
        else:
            gcs.append((pid, argv))
    if len(raylets) > 1:
        raise RuntimeError("Target pod must contain exactly one Raylet")
    if rank == 0:
        port = int(address.rsplit(":", 1)[1])
        owned = [pid for pid, argv in gcs if int(_flag(argv, "gcs_server_port")) == port]
        if len(owned) > 1:
            raise RuntimeError("Multiple GCS processes for the selected port")
        listener = _owns_listener(owned[0] if owned else None, port)
        if listener is False:
            raise RuntimeError("Selected GCS listener is not owned by head pod")
        if listener is None:
            return pending("gcs_listener_missing" if owned else "gcs_process_missing")
    if not raylets:
        return pending("raylet_process_missing")
    argv = raylets[0]
    python = shlex.split(_flag(argv, "python_worker_command"))[0]
    if not Path(python).is_absolute() or not Path(python).is_file():
        raise RuntimeError("Raylet must expose its actual Python executable")
    return {"ready": True, "node_id": _flag(argv, "node_id"), "python": python}


def validate_membership(expected: list[str], nodes: list[dict], gpus: int) -> bool:
    """拒绝外来/重复 NodeID 或 GPU 数不符；仅缺少预期节点时允许继续等候。"""
    if len(set(expected)) != len(expected):
        raise RuntimeError("Different pods reported the same Ray NodeID")
    alive = [node for node in nodes if node["Alive"]]
    actual = [node["NodeID"] for node in alive]
    if len(set(actual)) != len(actual):
        raise RuntimeError(f"Foreign or duplicate Ray nodes: expected={expected}, actual={actual}")
    if set(actual) - set(expected):
        raise RuntimeError(f"Foreign or duplicate Ray nodes: expected={expected}, actual={actual}")
    if any(node["Resources"].get("GPU", 0) != gpus for node in alive):
        raise RuntimeError("Ray node GPU allocation differs from trusted topology")
    return set(actual) == set(expected)


def main() -> None:
    """执行短时只读检查；失败直接非零退出，由编排层归为 INFRA。"""
    operation, target = sys.argv[1:]
    if operation == "allocate":
        result = allocated_port(int(target))
    elif operation == "member":
        result = local_member(target, int(os.environ["HARBOR_SERVE_RANK"]))
    elif operation == "cluster":
        target = validate_ray_address(target)
        os.environ["RAY_ADDRESS"] = target
        import ray

        ray.init(address=target, logging_level="ERROR", log_to_driver=False)
        try:
            result = {"nodes": ray.nodes()}
        finally:
            ray.shutdown()
    else:
        raise ValueError(f"Unknown isolation operation: {operation}")
    print("HARBOR_RAY_ISOLATION=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
