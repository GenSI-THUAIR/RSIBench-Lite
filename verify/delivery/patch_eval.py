"""patch 型评测链路的 task 无关公共件。

`InferVerifier`(推理题)与 `TrainVerifier`(训练题)共用同一套 patch 协议：
agent job 只产出引擎源码树的 Git patch，评测方在全新 job 里校验基线、应用
patch、执行 `<engine_repo>/launch.sh` 拉起被测服务，再从另一个**未打 patch**
的干净 job 用可信脚本打分。本模块只放这条协议本身，不含任何 task、引擎或
workload 的知识；那些差异一律由 task 的 `tests/verifier-config.json` 表达。

示例：
    endpoint = await start_patched_service(env, python="python3", ...)
    await wait_for_http_ready(endpoint, "/health", 1800, logger, "SGLang")
"""

from __future__ import annotations

from verify.resources import payload_file as verifier_script
import asyncio
import hashlib
import http.client
import json
import logging
import re
import shlex
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol

from harbor_envs.abstract_mlplat.retry import TransientError, retry_async

METADATA_NAME = "patch-metadata.json"
CONFIG_NAME = "verifier-config.json"
CONTROLLER_NAME = "serve_controller.py"
OCI_LAYER_NAME = "oci_layer.py"
REMOTE_VERIFIER_DIR = "/logs/verifier"
PORT_PROBE_NAME = "serve_port_probe.py"
# serve_controller._launch 落盘的 head 节点 launch 进程 pid 与服务 stdout。
SERVE_PID_PATH = "/workspace/deploy/rank-0.pid"
SERVE_HEAD_LOG_PATH = "/workspace/deploy/rank-0.log"


async def upload_trusted_bundle(environment, files: list[Path], output_dir: Path, remote: str) -> None:
    """压缩传输版本化可信脚本，并在 clean Driver 解包后逐文件核验 SHA。"""
    names = [path.name for path in files]
    if (not files or len(names) != len(set(names))
            or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names)
            or set(names) & {"record-sources.sha256", "record-sources.tar.gz"}
            or any(path.is_symlink() or not path.is_file() for path in files)):
        raise ValueError("可信脚本必须为名称唯一的普通文件")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "record-sources.sha256"
    manifest.write_text("".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in files))
    bundle = output_dir / "record-sources.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in [*files, manifest]:
            archive.add(path, arcname=path.name, recursive=False)
    await environment.upload_file(bundle, f"{remote}/record-sources.tar.gz")
    result = await environment.exec(
        f'tar -xzf {shlex.quote(remote + "/record-sources.tar.gz")} -C {shlex.quote(remote)} '
        f"&& cd {shlex.quote(remote)} && sha256sum -c record-sources.sha256",
        cwd="/tmp", timeout_sec=30,
    )
    if result.return_code != 0:
        raise RuntimeError(f"可信脚本解包/哈希校验失败：{result.stderr or result.stdout}")


class ControllerExitError(RuntimeError):
    """远端 controller 命令明确返回非零；不包含上传、拓扑或传输异常。"""


class ServiceEndpointError(RuntimeError):
    """端点身份或探针无法核验；不属于已确认的候选正确性失败。"""


class ServiceNotListening(ServiceEndpointError):
    """readiness 后服务已无监听，不能报告为其他 job 抢占。"""


class PortProbeError(ServiceEndpointError):
    """PID 证据缺失或探针失败；归属未知。"""


class ForeignPortOwner(ServiceEndpointError):
    """服务端口的监听者不是本 trial 的 launch 进程树。

    Merlin GPU pod 用宿主机网络，端口是整台机器共享的资源；health/readiness
    只证明「有服务在响应」，不证明是我们的服务在响应。
    """


class PatchEnvironment(Protocol):
    """本模块用到的 Harbor environment 子集(便于离线测试替身)。"""

    async def exec(
        self,
        command: str,
        cwd: str = ...,
        timeout_sec: float = ...,
        user: str | int | None = ...,
        pod_name: str | None = ...,
    ): ...

    async def upload_file(self, source: Path, destination: str) -> None: ...

    async def download_file(self, source: str, destination: Path) -> None: ...

    def role_pod_names(self) -> list[str]: ...

    async def service_addresses(self, port: int, *, pod_name: str | None = None): ...


def submission_artifact(raw: Mapping[str, object]) -> str:
    """从 verifier-config 取交付物文件名:layer_artifact(OCI layer,默认)或
    patch_artifact(legacy Git patch),必须恰有其一。"""
    layer = raw.get("layer_artifact")
    patch = raw.get("patch_artifact")
    if bool(layer) == bool(patch):
        raise ValueError("verifier-config 必须且只能声明 layer_artifact 或 patch_artifact 之一")
    return str(layer or patch)


def resolve_runtime_adapter(tests_dir: Path, name: object) -> Path | None:
    """解析 scoring 的 tests 相对 Python 文件；拒绝越界、符号链接及非文件。"""
    if name is None:
        return None
    if not isinstance(name, str) or not name:
        raise TypeError("runtime_adapter 必须是非空 tests 相对路径或 null")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
        raise ValueError("runtime_adapter 必须是 tests 内的相对 .py 文件")
    root = tests_dir.resolve(strict=True)
    target = root
    for part in relative.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError("runtime_adapter 路径不允许符号链接")
    target = target.resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("runtime_adapter 不在可信 tests 文件范围内")
    return target


async def wait_for_http_ready(
    endpoint: str,
    path: str,
    timeout_sec: int,
    logger: logging.Logger,
    label: str,
    accept_statuses: Iterable[int] = (200,),
    liveness_check: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """从 Harbor 宿主机轮询被测服务的就绪端点。

    `accept_statuses` 让不同引擎表达各自的就绪语义：SGLang/vLLM 的
    `/health` 返回 200，而 Tinker API server 只要求「HTTP 栈已经起来」，
    可以接受 404/405。
    """
    accepted = frozenset(accept_statuses)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    deadline = time.monotonic() + timeout_sec

    def probe() -> bool:
        request = urllib.request.Request(f"{endpoint}{path}")
        try:
            with opener.open(request, timeout=max(0.001, min(20, deadline - time.monotonic()))) as response:
                return response.status in accepted
        except urllib.error.HTTPError as error:
            return error.code in accepted
        except (OSError, http.client.HTTPException):
            # 端口被同宿主机其它进程(如 raylet idle worker)抢占时会收到非 HTTP 的二进制响应
            # (BadStatusLine)：按未就绪继续轮询/超时，由 launch 存活看门狗与端口归属校验兜底定性。
            return False

    try:
        async with asyncio.timeout(timeout_sec):
            attempt = 0
            while time.monotonic() < deadline:
                attempt += 1
                if liveness_check is not None:
                    await liveness_check()
                if await asyncio.to_thread(probe):
                    return
                if attempt % 6 == 0:
                    logger.info("等待 patched %s 服务，剩余 %.1fs", label, deadline - time.monotonic())
                await asyncio.sleep(min(10, max(0, deadline - time.monotonic())))
    except TimeoutError as exc:
        raise RuntimeError(f"patched 服务 {timeout_sec}s 内未就绪：{endpoint}{path}") from exc
    raise RuntimeError(f"patched 服务 {timeout_sec}s 内未就绪：{endpoint}{path}")


async def _ray_probe(environment, pod, operation, target, *, python="python3", rank=0,
                     timeout_sec=30):
    """显式 pod 短探针；调用方剩余预算约束等待，远端 CLI 使用整数秒超时。"""
    async with asyncio.timeout(timeout_sec):
        result = await environment.exec(
            f"env HARBOR_SERVE_RANK={rank} RAY_ADDRESS={shlex.quote(str(target))} "
            f"{shlex.quote(python)} {REMOTE_VERIFIER_DIR}/ray_isolation.py "
            f"{operation} {shlex.quote(str(target))}",
            # CLI 只接受整数秒；不足一秒的尾部仍由本地 asyncio 精确限时。
            cwd="/tmp", timeout_sec=max(1, int(min(30, timeout_sec))), pod_name=pod,
        )
    if result.return_code != 0:
        raise RuntimeError(f"Ray isolation {operation} failed on {pod}: {result.stderr or result.stdout}")
    records = [line.removeprefix("HARBOR_RAY_ISOLATION=")
               for line in (result.stdout or "").splitlines()
               if line.startswith("HARBOR_RAY_ISOLATION=")]
    if len(records) != 1:
        raise RuntimeError(f"Ray isolation {operation} missing unique result on {pod}")
    return json.loads(records[0])


async def _wait_for_ray_membership(environment, pods, address, gpus, timeout_sec):
    """共享总截止时间；只重试明确的启动未就绪，任何归属异常立即失败。"""
    from verify.resources import payload_module
    validate_membership = payload_module("ray_isolation.py").validate_membership

    deadline = time.monotonic() + timeout_sec

    def remaining():
        """返回唯一总预算的剩余时间；探针前后、成功前均拒绝超时。"""
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Ray isolation deadline exceeded")
        return value

    try:
        async with asyncio.timeout(timeout_sec):
            while True:
                members = []
                for rank, pod in enumerate(pods):
                    member = await _ray_probe(environment, pod, "member", address,
                                              rank=rank, timeout_sec=min(30, remaining()))
                    remaining()
                    if member["ready"] is not True:
                        if member["ready"] is not False or member.get("reason") not in {
                            "gcs_process_missing", "gcs_listener_missing", "raylet_process_missing",
                        }:
                            raise RuntimeError("Unrecognized Ray startup-not-ready result")
                    members.append(member)
                if all(member["ready"] for member in members):
                    cluster = await _ray_probe(environment, pods[0], "cluster", address,
                                               python=members[0]["python"],
                                               timeout_sec=min(30, remaining()))
                    remaining()
                    if validate_membership([member["node_id"] for member in members],
                                           cluster["nodes"], gpus):
                        remaining()
                        return
                await asyncio.sleep(min(5, remaining()))
    except TimeoutError as exc:
        raise RuntimeError("Ray isolation timed out before complete pod attestation") from exc


async def start_patched_service(
    environment: PatchEnvironment,
    *,
    python: str,
    patch_artifact: str,
    patch_path: Path,
    metadata_path: Path,
    config_path: Path,
    port: int,
    timeout_sec: float = 1200,
    ray_isolation: bool = False,
    runtime_adapter: Path | None = None,
    evaluation_baseline_dir: Path | None = None,
) -> str:
    """上传可信 controller，在干净 job 应用 patch 并启动，返回服务地址。

    ray_isolation=True 是显式启动契约：candidate 已支持 RAY_PORT/RAY_ADDRESS，
    环境能提供真实 head_pod 与平台 PORT<n>，并使用冻结的 Ray 2.56 CLI
    默认端口契约；不满足则 INFRA，不改写 candidate 或 launch 参数。
    默认不启用，以免静默改变既有实验；调用方必须对新实验显式选择该契约。
    """
    config = json.loads(config_path.read_text())
    baseline = config.get('evaluation_baseline')
    if baseline is not None:
        from verify.delivery.evaluation_baseline import build
        if evaluation_baseline_dir is None:
            raise ValueError('Evaluation baseline requires trusted task environment directory')
        manifest = config_path.with_suffix('.baseline.json')
        manifest.write_text(json.dumps(build(evaluation_baseline_dir, baseline)) + '\n')
    node_count = int(config["serve_node_count"])
    adapter_args = ""
    if runtime_adapter is not None:
        if not isinstance(runtime_adapter, Path) or not runtime_adapter.is_absolute():
            raise TypeError("runtime_adapter 必须是可信调用方提供的绝对 Path")
        if runtime_adapter.is_symlink() or not runtime_adapter.is_file() or runtime_adapter.suffix != ".py":
            raise ValueError("runtime_adapter 必须是非符号链接 Python 文件")
        adapter_sha = hashlib.sha256(runtime_adapter.read_bytes()).hexdigest()
        adapter_args = (f" --runtime-adapter {REMOTE_VERIFIER_DIR}/runtime_adapter.py"
                        f" --runtime-adapter-sha256 {adapter_sha}")

    # 逐 pod 路径不依赖平台预起 Ray，避免 robust entrypoint 缺失 sshd。
    pods: list[str | None] = [None]
    if node_count > 1 or ray_isolation:
        names = environment.role_pod_names()
        if len(names) < node_count:
            raise RuntimeError(
                f"job 只有 {len(names)} 个 pod，不足 serve_node_count={node_count}"
            )
        pods = list(names[:node_count])
        if ray_isolation:
            head = getattr(environment, "head_pod", None)
            if not head or head not in names or len(set(names)) != node_count or len(names) != node_count:
                raise RuntimeError("Ray isolation requires the actual head and exact platform pod set")
            pods = [head, *(pod for pod in names if pod != head)]
            if {"RAY_PORT", "RAY_ADDRESS"} & config.get("launch_env", {}).keys():
                raise RuntimeError("Static launch_env must not override trusted Ray port/address")

    # /logs 为 pod 本地目录，每个节点都上传可信 controller 与冻结输入。
    for pod in pods:
        if pod is not None:
            prepared = await environment.exec(
                f"mkdir -p {shlex.quote(REMOTE_VERIFIER_DIR)} "
                f"&& chmod 0777 {shlex.quote(REMOTE_VERIFIER_DIR)}",
                cwd="/tmp", timeout_sec=60, user="root", pod_name=pod,
            )
            if prepared.return_code != 0:
                raise RuntimeError(
                    f"pod {pod} 上无法创建 {REMOTE_VERIFIER_DIR}："
                    f"{prepared.stderr or prepared.stdout}"
                )
        await environment.upload_file(
            verifier_script(CONTROLLER_NAME), f"/logs/verifier/{CONTROLLER_NAME}", pod_name=pod,
        )
        await environment.upload_file(
            verifier_script(OCI_LAYER_NAME), f"/logs/verifier/{OCI_LAYER_NAME}", pod_name=pod,
        )
        await environment.upload_file(config_path, f"/logs/verifier/{CONFIG_NAME}", pod_name=pod)
        await environment.upload_file(patch_path, f"/logs/verifier/{patch_artifact}", pod_name=pod)
        await environment.upload_file(metadata_path, f"/logs/verifier/{METADATA_NAME}", pod_name=pod)
        if baseline is not None:
            await environment.upload_file(manifest, f'{REMOTE_VERIFIER_DIR}/evaluation-baseline.json', pod_name=pod)
            await environment.upload_file(verifier_script('evaluation_baseline.py'), f'{REMOTE_VERIFIER_DIR}/evaluation_baseline.py', pod_name=pod)
        if runtime_adapter is not None:
            await environment.upload_file(runtime_adapter, f"{REMOTE_VERIFIER_DIR}/runtime_adapter.py", pod_name=pod)
        if ray_isolation:
            await environment.upload_file(verifier_script("ray_isolation.py"),
                                          f"{REMOTE_VERIFIER_DIR}/ray_isolation.py", pod_name=pod)

    addresses = await environment.service_addresses(port,pod_name=pods[0])
    head_ip, head_ipv4 = addresses.host, addresses.ipv4
    prefix = ""
    if ray_isolation:
        from verify.resources import payload_module
        RAY_PORT_CONTRACT = payload_module("ray_isolation.py").RAY_PORT_CONTRACT
        compatible_ray_port = payload_module("ray_isolation.py").compatible_ray_port

        allocation = await _ray_probe(environment, pods[0], "allocate", port)
        ray_port = allocation["port"]
        source = allocation.get("source", "")
        slot = source.removeprefix("head:PORT") if isinstance(source, str) else ""
        if (not isinstance(source, str) or not source.startswith("head:PORT")
                or not slot.isascii() or not slot.isdecimal()
                or allocation.get("contract") != RAY_PORT_CONTRACT
                or not compatible_ray_port(ray_port, port)):
            raise RuntimeError("Ray isolation received invalid platform port evidence")
        ray_address = f"{head_ipv4}:{ray_port}"
        prefix = (f"env RAY_PORT={ray_port} RAY_ADDRESS={shlex.quote(ray_address)} "
                  f"HARBOR_SERVE_HEAD_IPV4={shlex.quote(head_ipv4)} ")

    # rank 0 必须先拉起；所有 rank 只接收 head 选出的同一个 GCS 地址。
    for rank, pod in enumerate(pods):
        controller_command = (
            f"{shlex.quote(python)} /logs/verifier/{CONTROLLER_NAME} "
            f"/logs/verifier/{patch_artifact} /logs/verifier/{METADATA_NAME} "
            f"/logs/verifier/{CONFIG_NAME} "
            f"--rank {rank} --head-ip {shlex.quote(head_ip)} "
            f"--head-ipv4 {shlex.quote(head_ipv4)}{adapter_args}"
        )
        # 远端 shell 明确报出 controller 退出码，区分 SSH/CLI 自身失败。
        wrapped = (
            f"if {controller_command}; then rc=0; else rc=$?; fi; "
            "printf '\\nHARBOR_CONTROLLER_EXIT=%s\\n' \"$rc\"; exit \"$rc\""
        )
        result = await environment.exec(
            prefix + "bash -c " + shlex.quote(wrapped),
            cwd="/tmp", timeout_sec=timeout_sec, pod_name=pod,
        )
        if result.return_code != 0:
            markers = [line for line in (result.stdout or "").splitlines()
                       if line.startswith("HARBOR_CONTROLLER_EXIT=")]
            error_type = (ControllerExitError
                          if markers == [f"HARBOR_CONTROLLER_EXIT={result.return_code}"]
                          else RuntimeError)
            raise error_type(
                f"patched job rank {rank} 启动 controller 失败："
                f"{result.stderr or result.stdout}"
            )
        if runtime_adapter is not None:
            # download_file 没有 pod 参数；小证据通过已支持逐 pod 的 exec 读取。
            evidence_path = f"/workspace/deploy/runtime-adapter-rank-{rank}.json"
            read_evidence = (
                "import json;from pathlib import Path;"
                f"value=json.loads(Path({evidence_path!r}).read_text());"
                "print('HARBOR_RUNTIME_ADAPTER='+json.dumps(value,sort_keys=True))"
            )
            result = await environment.exec(
                f"{shlex.quote(python)} -c {shlex.quote(read_evidence)}",
                cwd="/tmp", timeout_sec=30, pod_name=pod,
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr or result.stdout or "adapter evidence read failed")
            frames = [line.removeprefix("HARBOR_RUNTIME_ADAPTER=")
                      for line in (result.stdout or "").splitlines()
                      if line.startswith("HARBOR_RUNTIME_ADAPTER=")]
            if len(frames) != 1:
                raise RuntimeError("runtime adapter evidence frame missing or ambiguous")
            evidence = json.loads(frames[0])
            if (evidence.get("schema") != "runtime-adapter-evidence/1"
                    or evidence.get("rank") != rank
                    or evidence.get("adapter_sha256") != adapter_sha):
                raise RuntimeError("runtime adapter evidence identity mismatch")
            config_path.with_name(f"{config_path.stem}-runtime-adapter-rank-{rank}.json").write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n"
            )
    if ray_isolation:
        await _wait_for_ray_membership(environment, pods, ray_address,
                                       int(config["serve_gpus_per_node"]), timeout_sec)
    return addresses.endpoint


async def assert_port_owned(
    environment: PatchEnvironment,
    logger: logging.Logger,
    *,
    endpoint: str = "",
    port: int,
    pid_path: str = SERVE_PID_PATH,
    log_path: str = SERVE_HEAD_LOG_PATH,
) -> None:
    """断言 head 节点上 *port* 的实际应答方属于本 trial 的服务进程树。

    必须在 health/readiness 通过之后调用。0902 实录（historical validation
    infer-llada2p1mini-16b）：本 trial 的 serve 实例 EADDRINUSE 秒退，health 200
    来自同宿主机另一 job 的 Qwen3 thinking 服务，gate/benchmark 全程给别人
    的模型打了分，verifier 却报 deploy_success=1。归属校验失败抛
    :class:`ForeignPortOwner`；只有明确的传输层抖动（ssh probe failed /
    本地 CLI 超时）做有限重试。

    判定细节在 verify/delivery/serve_port_probe.py：socket 表交叉 + nonce 端到端
    仲裁双层。*endpoint* 强烈建议传入——共享宿主机 netns 下 /proc/net 是
    全机表且单次读取可能漏行（0902 深夜首版探针据此误杀过健康部署），
    nonce 仲裁不依赖任何 /proc 可见性。

    契约约束：launch.sh 及其服务进程需保持前台进程树（现有全部 task 都是
    `exec` 前台常驻）且服务把 access log 写到 stdout（uvicorn/sglang 默认
    如此）；两者都不满足的引擎会被误判，届时探针的 DIAG 输出会直接指出。
    """
    await environment.upload_file(
        verifier_script(PORT_PROBE_NAME),
        f"{REMOTE_VERIFIER_DIR}/{PORT_PROBE_NAME}",
    )
    arbitration = ""
    if endpoint:
        arbitration = (
            f" --endpoint {shlex.quote(endpoint)} --log {shlex.quote(log_path)}"
        )

    async def _probe():
        try:
            result = await environment.exec(
                f"python3 {REMOTE_VERIFIER_DIR}/{PORT_PROBE_NAME} "
                f"{port} {shlex.quote(pid_path)}{arbitration}",
                cwd="/tmp",
                timeout_sec=180,
                user="root",
            )
        except subprocess.TimeoutExpired as exc:
            raise TransientError(f"port probe local CLI timeout: {exc}") from exc
        if result.return_code != 0 and "ssh probe failed" in (result.stderr or ""):
            raise TransientError((result.stderr or "").strip()[-300:])
        return result

    result = await retry_async(
        _probe,
        what=f"probe port {port} ownership",
        attempts=3,
        delay_sec=10.0,
        backoff=1.0,
        retry_on=(TransientError,),
        logger=logger,
    )
    if result.return_code != 0:
        error_type = {2: ServiceNotListening, 3: ForeignPortOwner}.get(
            result.return_code, PortProbeError
        )
        raise error_type(
            f"服务端口归属校验失败（不保证来自本 trial，同宿主机端点需核验；rc={result.return_code}）："
            f"{(result.stdout or result.stderr or '').strip()[:900]}"
        )


async def verify_clean_baseline(
    environment: PatchEnvironment,
    engine_repo: str,
    metadata: Mapping[str, object],
) -> None:
    """确认打分 job 的引擎源码树与 agent 启动前的基线逐字节一致。

    打分 job 不打 patch,所以 agent 对 benchmark/driver 脚本的任何改动都
    进不了计分链路;这里的两道校验(HEAD commit + 完整 worktree tree)是该
    保证的执行点。
    """
    repo = shlex.quote(engine_repo)
    commit_result = await environment.exec(
        f"sudo -n git -C {repo} rev-parse HEAD", cwd="/tmp", timeout_sec=30
    )
    actual_commit = (commit_result.stdout or "").strip()
    if commit_result.return_code != 0 or actual_commit != metadata["base_commit"]:
        raise RuntimeError(
            "打分 job 引擎基线不一致："
            f"expected={metadata['base_commit']}, actual={actual_commit}"
        )
    tree_result = await environment.exec(
        "set -e\n"
        "index=/tmp/.harbor-benchmark-baseline.index\n"
        "sudo -n rm -f $index\n"
        f"sudo -n env GIT_INDEX_FILE=$index git -C {repo} read-tree HEAD\n"
        f"sudo -n env GIT_INDEX_FILE=$index git -C {repo} add -A -- .\n"
        f"sudo -n env GIT_INDEX_FILE=$index git -C {repo} write-tree\n"
        "sudo -n rm -f $index\n",
        cwd="/tmp",
        timeout_sec=300,
    )
    tree_lines = (tree_result.stdout or "").strip().splitlines()
    actual_tree = tree_lines[-1] if tree_lines else ""
    if tree_result.return_code != 0 or actual_tree != metadata["baseline_tree"]:
        raise RuntimeError(
            "打分 job 引擎初始 worktree 不一致："
            f"expected={metadata['baseline_tree']}, actual={actual_tree}"
        )


async def raise_if_environment_stopped(environment, *, context: str) -> None:
    """只读检查权威终态；观察超时不是停止，也不授予任何资源操作权限。"""
    query = getattr(environment, 'terminal_state_reason', None)
    if query is None:
        return
    try:
        reason = await query()
    except (subprocess.TimeoutExpired, TimeoutError):
        return
    if reason is not None:
        raise RuntimeError(f'{context} aborted: {reason}')


async def run_detached(
    environment: PatchEnvironment,
    *,
    command: str,
    tag: str,
    logs_dir: str = "/logs/verifier",
    poll_interval_sec: float = 60,
    timeout_sec: float = 7200,
    dependencies: tuple[PatchEnvironment, ...] = (),
    poll_status: Callable[[str], Awaitable[object]] | None = None,
) -> int:
    """以 detached 进程执行长任务，用 rc 文件轮询结果，返回退出码。

    WebShell 对长时间持有的单条 exec 不可靠(iter-v30/31/34:40min 的
    benchmark exec 之后通道死透,首个下载即毁掉整个 verifier)。所有超过
    几分钟的可信脚本都必须走这个模式。

    environment 可选提供 async terminal_state_reason() -> str | None：仅明确
    平台终态中止等待，查询暂态返回 None；旧 PodView/最小 exec 替身无需实现。
    轮询与状态查询共用总期限；这里不执行 stop/delete，也不切换 SSH backend。
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError("Invalid detached tag")
    prefix = f"{logs_dir}/{tag}"
    token = uuid.uuid4().hex
    rc_path = f"{prefix}.rc"
    identity = shlex.quote(f"{prefix}.{token}.identity")
    cancelled = shlex.quote(f"{prefix}.{token}.cancel")
    # wrapper 自身也必须 detached；原版前台 wait 会被启动 exec 的 60s 超时杀掉。
    wrapper = (
        'read -r stat < /proc/$$/stat; stat=${stat##*) }; fields=($stat)\n'
        f'printf "%s %s %s %s %s\\n" "$$" "${{fields[2]}}" "${{fields[3]}}" "${{fields[19]}}" {token} > {identity}\n'
        f"test ! -e {cancelled} || exit 125\n"
        f"if {command}; then rc=0; else rc=$?; fi\n"
        f'printf "%s %s\\n" "$rc" {token} > {shlex.quote(rc_path + "." + token + ".tmp")}\n'
        f"mv {shlex.quote(rc_path + '.' + token + '.tmp')} {shlex.quote(rc_path)}"
    )
    launch_command = (
        "bash -lc "
        + shlex.quote(
            f"set -e\ntest ! -e {cancelled} || exit 125\n"
            f"setsid nohup env HARBOR_DRIVER_TOKEN={token} bash -c {shlex.quote(wrapper)} "
            f"> {shlex.quote(prefix + '.log')} 2>&1 < /dev/null &\n"
            f"printf '%s\\n' $! > {shlex.quote(prefix + '.pid')}"
        )
    )
    terminal_state_reason = getattr(environment, "terminal_state_reason", None)
    job_terminal = False
    try:
        async with asyncio.timeout(timeout_sec):
            # 启动 SSH 也消耗同一总预算；超时回执不表示远端没有迟到启动。
            launch = await environment.exec(launch_command, cwd=logs_dir, timeout_sec=60)
            if launch.return_code != 0:
                raise RuntimeError(f"{tag} launch failed: {launch.stderr or launch.stdout}")
            while True:
                await asyncio.sleep(poll_interval_sec)
                for dependency in dependencies:
                    await raise_if_environment_stopped(dependency, context=tag)
                try:
                    poll_command = "bash -lc " \
                        + shlex.quote(
                            f"if test -f {shlex.quote(rc_path)}; then "
                            f"read -r rc token < {shlex.quote(rc_path)}; "
                            f'if [[ "$token" = {token} ]]; then echo DONE "$rc"; else echo RUNNING; fi; '
                            "else echo RUNNING; fi"
                        )
                    if poll_status is None:
                        poll = await environment.exec(poll_command, cwd="/tmp", timeout_sec=120)
                    else:
                        poll = await poll_status(poll_command)
                except (subprocess.TimeoutExpired, TimeoutError):
                    # SSH/本地 CLI 超时不是 job 终态；仍查询控制面再有界重试。
                    poll = None
                if poll is not None and poll.return_code == 0:
                    state = (poll.stdout or "").strip().splitlines()
                    marker = next(
                        (line for line in reversed(state) if line.startswith(("DONE", "RUNNING"))),
                        "",
                    )
                    if marker.startswith("DONE"):
                        fields = marker.split()
                        if len(fields) != 2 or not fields[1].isdigit() or not 0 <= int(fields[1]) <= 255:
                            raise RuntimeError(f"{tag} invalid terminal receipt: {marker}")
                        return int(fields[1])
                if terminal_state_reason is not None:
                    reason = await terminal_state_reason()
                    if reason is not None:
                        job_terminal = True
                        raise RuntimeError(f"{tag} aborted: {reason}")
    except TimeoutError as exc:
        await _stop_detached_driver(environment, prefix, token)
        raise RuntimeError(f"{tag} 在 {timeout_sec}s 内未结束") from exc
    except BaseException:
        if not job_terminal:
            await _stop_detached_driver(environment, prefix, token)
        raise


async def _stop_detached_driver(environment, prefix: str, token: str) -> None:
    """只清理该 invocation 登记且 birth/PGID/SID 仍一致的进程组。"""
    command = f"""touch {shlex.quote(prefix + '.' + token + '.cancel')} || exit 1
test -s {shlex.quote(prefix + '.' + token + '.identity')} || exit 0
read -r pid pgid sid birth token < {shlex.quote(prefix + '.' + token + '.identity')}
[[ "$token" = {shlex.quote(token)} && "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]] || exit 0
[[ "$pid" = "$pgid" && "$pid" = "$sid" ]] || exit 0
matches() {{
  local stat
  local -a fields
  test -r /proc/$pid/stat || return 1
  read -r stat < /proc/$pid/stat || return 1
  stat=${{stat##*) }}; fields=($stat)
  [[ "${{fields[2]}} ${{fields[3]}} ${{fields[19]}}" = "$pgid $sid $birth" ]]
}}
# TERM 可能先结束 leader；记录现存组员的 birth，随后只清理这些仍匹配的进程。
members=()
for entry in /proc/[0-9]*/stat; do
  read -r stat < "$entry" 2>/dev/null || continue
  member=${{entry#/proc/}}; member=${{member%/stat}}
  stat=${{stat##*) }}; fields=($stat)
  if [[ "${{fields[2]}} ${{fields[3]}}" = "$pgid $sid" ]]; then
    if matches || grep -zqxF "HARBOR_DRIVER_TOKEN=$token" /proc/$member/environ 2>/dev/null; then
      members+=("$member:${{fields[19]}}")
    fi
  fi
done
(( ${{#members[@]}} )) || exit 0
kill -TERM -- -$pgid 2>/dev/null || true
sleep 2
for entry in "${{members[@]}}"; do
  member=${{entry%:*}}; expected=${{entry#*:}}
  read -r stat < /proc/$member/stat 2>/dev/null || continue
  stat=${{stat##*) }}; fields=($stat)
  if [[ "${{fields[2]}} ${{fields[3]}} ${{fields[19]}}" = "$pgid $sid $expected" ]]; then
    kill -KILL "$member" 2>/dev/null || true
  fi
done
"""
    try:
        async with asyncio.timeout(10):
            result = await environment.exec("bash -c " + shlex.quote(command), cwd="/tmp", timeout_sec=5)
        if result.return_code != 0:
            logging.getLogger(__name__).warning("driver cleanup unconfirmed for %s: %s", prefix, result.stderr)
    except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
        logging.getLogger(__name__).warning("driver cleanup unconfirmed for %s: %s", prefix, exc)


async def tail_log(
    environment: PatchEnvironment, tag: str, logs_dir: str = "/logs/verifier"
) -> str:
    """取 detached 任务日志尾部，用于失败时的诊断信息。"""
    tail = await environment.exec(
        f"tail -c 3000 {logs_dir}/{tag}.log", cwd="/tmp", timeout_sec=120
    )
    return tail.stdout or tail.stderr or ""


def load_json_artifact(path: Path) -> Mapping[str, object]:
    """读取可信脚本产出的 JSON object;类型不对一律 fail loud。"""
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} 必须是 JSON object")
    return value
