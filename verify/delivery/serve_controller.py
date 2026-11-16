"""在全新 Merlin job 中应用引擎源码 patch 并启动服务。

参数：patch_path 为 Git patch；metadata_path 为包含 base_commit 和 sha256 的
JSON；config_path 为 task 的 verifier-config.json(提供 engine_repo 与节点
拓扑)。patch 应用目标是 config 的 engine_repo 指向的源码树(SGLang task 为
/sgl-workspace/sglang,vLLM-Ascend task 为 /vllm-workspace/vllm-ascend),
启动合约为该树根目录下的 launch.sh。

两条启动路径同时成立,按调用姿势选择:

* **拓扑自动路径**(推理题,infer_verifier 的旧 CLI):单节点
  (serve_node_count=1)直接在本机应用 patch 并启动,不依赖 Ray;多节点走
  平台预起的 Ray,detached actor 按节点放置。支持 `--relaunch`(部署重试)
  与 `--oracle`(标定基线)。
* **逐 pod 路径**(训练题,patch_eval/train_verifier 的新 CLI):显式
  `--rank` 时只管**本 pod**,由宿主机逐 pod 调用。平台的
  `enable_ray_on_arnold` 在多节点时会把 job 切到 Arnold 的 robust-training
  entrypoint,那条路径**不配置 Merlin 的 sshd**,Harbor 的 SSH exec 因此
  永远连不上;训练题 job 关掉该开关,rank 与 head 地址经环境变量
  `HARBOR_SERVE_RANK` / `RAY_IP` 透给 launch.sh,由它决定本节点是
  `ray start --head` 还是 `ray start --address`。serve_controller 自身
  保持引擎无关。

输出：各节点(或本节点)的 patch、进程和日志状态(JSON 数组)。

示例：
    python3 serve_controller.py \
        /logs/verifier/sglang.patch /logs/verifier/patch-metadata.json \
        /logs/verifier/verifier-config.json
    python3 serve_controller.py \
        /logs/verifier/skyrl.patch /logs/verifier/patch-metadata.json \
        /logs/verifier/verifier-config.json --rank 1 --head-ip 2605:340::1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

WORKDIR = Path("/workspace/deploy")
# 与本脚本一起上传到每个 serve pod 的 OCI layer 应用器(见 verify.delivery.patch_eval)。
OCI_LAYER_SCRIPT = "/logs/verifier/oci_layer.py"
OCI_LAYER_TMP = Path("/tmp/harbor-submit_oci.tar.gz")
EVALUATION_BASELINE_PATH = Path('/logs/verifier/evaluation-baseline.json')
NAMESPACE = "harbor-patched-serve"
ACTOR_PREFIX = "patched-serve-rank-"


def launch_wrapper_command(launch_script: Path, rc_path: Path) -> str:
    """构造启动包装命令;输入 launch 脚本与 rc 路径,输出 bash -c 命令体。

    外层 bash 在 launch.sh(含其 exec 的服务进程)退出后写 rc 文件;服务
    健康时常驻进程不退出,rc 永不出现。verifier 的 health 轮询据此对已死
    服务 fail-fast(v19:服务启动 3 分钟即死,health poll 仍空等满 5400s)。
    """
    return (
        f"bash {shlex.quote(str(launch_script))}; "
        f'echo "$?" > {shlex.quote(str(rc_path))}'
    )


def load_serve_config(config_path: Path) -> tuple[str, int, int]:
    """从 verifier-config.json 读取引擎源码树路径、节点数和每节点卡数。"""
    config = json.loads(config_path.read_text())
    engine_repo = str(config["engine_repo"])
    node_count = int(config["serve_node_count"])
    gpus_per_node = int(config["serve_gpus_per_node"])
    if not engine_repo.startswith("/"):
        raise ValueError(f"engine_repo 必须是绝对路径：{config_path}")
    if node_count <= 0 or gpus_per_node <= 0:
        raise ValueError(
            f"serve_node_count/serve_gpus_per_node 必须为正数：{config_path}"
        )
    return engine_repo, node_count, gpus_per_node


def ray_address() -> str:
    """根据 Arnold 环境变量构造显式 Ray GCS 地址。"""
    ip = os.environ.get("RAY_IP") or os.environ.get("BYTED_RAY_POD_IP")
    port = os.environ.get("RAY_SERVE_PORT") or os.environ.get("PORT5")
    if not ip or not port:
        raise RuntimeError(
            "RAY_IP/BYTED_RAY_POD_IP 和 RAY_SERVE_PORT/PORT5 必须存在"
        )
    host = f"[{ip}]" if ":" in ip and not ip.startswith("[") else ip
    return f"{host}:{port}"


def wait_for_ray_gcs() -> str:
    """等待 Arnold 完成挂载并启动 Ray GCS，返回显式连接地址(多节点路径)。"""
    ip = os.environ.get("RAY_IP") or os.environ.get("BYTED_RAY_POD_IP")
    port = os.environ.get("RAY_SERVE_PORT") or os.environ.get("PORT5")
    if not ip or not port:
        raise RuntimeError(
            "RAY_IP/BYTED_RAY_POD_IP 和 RAY_SERVE_PORT/PORT5 必须存在"
        )
    for attempt in range(1, 91):
        try:
            with socket.create_connection((ip, int(port)), timeout=5):
                return ray_address()
        except OSError:
            if attempt % 6 == 0:
                print(f"等待 Ray GCS：{ip}:{port}（{attempt}/90）", flush=True)
            time.sleep(10)
    raise RuntimeError(f"15 分钟内 Ray GCS 未就绪：{ip}:{port}")


def worktree_tree(repo: Path, index: int) -> str:
    """用独立 Git index 计算当前完整 worktree tree，不改变仓库 index。"""
    index_path = f"/tmp/.harbor-serve-baseline-{os.getpid()}-{index}.index"
    subprocess.run(["sudo", "-n", "rm", "-f", index_path], check=True)
    prefix = ["sudo", "-n", "env", f"GIT_INDEX_FILE={index_path}"]
    try:
        subprocess.run(
            [*prefix, "git", "-C", str(repo), "read-tree", "HEAD"], check=True
        )
        subprocess.run(
            [*prefix, "git", "-C", str(repo), "add", "-A", "--", "."],
            check=True,
        )
        return subprocess.check_output(
            [*prefix, "git", "-C", str(repo), "write-tree"], text=True
        ).strip()
    finally:
        subprocess.run(["sudo", "-n", "rm", "-f", index_path], check=True)


class ServiceRunner:
    """在固定节点应用 patch，并持有该节点的服务进程。

    单节点路径直接实例化本类;多节点路径由 main() 包装成 Ray detached
    actor(`ray.remote(ServiceRunner)`)按节点放置。"""

    def __init__(self, index: int, engine_repo: str):
        self.index = index
        self.repo = Path(engine_repo)
        self.launch_script = self.repo / "launch.sh"
        self.process: subprocess.Popen[bytes] | None = None
        self.log_file = None
        self.log_path = WORKDIR / f"rank-{index}.log"
        # launch 退出码落盘,verifier 的 health 轮询据此对已死服务 fail-fast
        # (v19:服务启动 3 分钟即死,health poll 仍空等满 5400s)。
        self.rc_path = WORKDIR / f"rank-{index}.rc"
        # 宿主机侧的看门狗靠这个文件判断本节点服务是否还活着。**必须用 pid 而
        # 不是进程名**:launch.sh 结尾通常是 `exec <真正的服务进程>`(rank 0 exec
        # 成 server，其余 rank exec 成常驻等待)，exec 之后 cmdline 就不再是
        # `bash .../launch.sh` 了，按名字 pgrep 一定找不到;而 exec **不改 pid**，
        # 所以 `kill -0 <pid>` 对两种 rank 都成立。按名字 pgrep 还有第二个坑:
        # 远端是 `bash -c '… pgrep -f "bash .../launch.sh" …'`，模式串在执行
        # 命令自己的 cmdline 里，pgrep 会 self-match 从而永远报「活着」——
        # 那等于看门狗完全失效(本仓库已在别处踩过同一个坑)。
        self.pid_path = WORKDIR / f"rank-{index}.pid"
        self.runtime_adapter_evidence: dict[str, object] | None = None

    def apply_and_start(
        self,
        patch: bytes,
        expected_sha256: str,
        base_commit: str,
        baseline_tree: str,
        head_ip: str,
        relaunch: bool = False,
        oracle_script: str = "",
        runtime_adapter: tuple[Path, str] | None = None,
        expected_patch_tree: str = "",
        oci: dict[str, object] | None = None,
        evaluation_baseline: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """校验基线和交付物(OCI layer 或 legacy patch),应用后执行仓库内的 launch.sh。

        oci 非 None 即 OCI layer 模式(默认协议):patch 参数携带的是
        submit_oci.tar.gz 的字节,经 sha256 校验后交给 oci_layer.py 以 root 叠到 /,
        并核对其回报的 layer_sha256/diff_id;oci["diff_id"] 来自 collect 元数据,
        oci["forbidden_paths"] 来自可信 verifier-config(只能追加禁止路径)。

        relaunch=True 用于部署重试(如 Ascend aclgraph 捕获偶发竞态,
        regrade3 实测 error 107025):patch 已应用,跳过基线/patch 校验,
        清掉残余引擎进程后直接重启 launch.sh。

        oracle_script 非空则进入 oracle 基线模式:在本节点执行该脚本
        (契约是产出 repo 内的 launch.sh,如把 task 的 environment/launch.sh 装进
        仓库的 baseline.sh)后启动服务。
        标定 oracle 没有 patch,基线/tree/sha256 三项校验是为「验证 agent 交付
        物」设计的,对 oracle 无意义——此前为迁就 API 伪造空/合成 patch,连带
        踩了 tree 顺序、全仓 diff 卷入镜像预存未跟踪文件等多轮失败。
        """
        if runtime_adapter is not None and (oracle_script or relaunch):
            raise ValueError("runtime_adapter 仅支持完整 patch 校验路径，不支持 oracle/relaunch")
        if oracle_script:
            script = Path(oracle_script)
            if not script.is_file():
                raise FileNotFoundError(f"oracle 脚本不存在：{script}")
            subprocess.run(
                ["sudo", "-n", "bash", str(script)], cwd=self.repo, check=True
            )
            subprocess.run(
                ["sudo", "-n", "chmod", "+x", str(self.launch_script)], check=True
            )
            # 与 patch 路径同款:solve.sh 以 root 写入,launch 以当前用户执行,
            # repo 内 root 属主目录会让引擎写不进(r5 实测 _cann_ops_custom)。
            subprocess.run(
                ["sudo", "-n", "chown", "-R",
                 f"{os.getuid()}:{os.getgid()}", str(self.repo)],
                check=True,
            )
            return self._launch(head_ip)
        if relaunch:
            subprocess.run(
                ["sudo", "-n", "pkill", "-9", "-f",
                 r"[s]glang.launch_server|[v]llm serve|[A]PIServer"],
                check=False,
            )
            time.sleep(5)
            return self._launch(head_ip)
        actual_commit = subprocess.check_output(
            ["sudo", "-n", "git", "-C", str(self.repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if actual_commit != base_commit:
            raise RuntimeError(
                f"引擎基线不一致：expected={base_commit}, actual={actual_commit}"
            )
        actual_tree = worktree_tree(self.repo, self.index)
        if actual_tree != baseline_tree:
            raise RuntimeError(
                "引擎初始 worktree 不一致："
                f"expected={baseline_tree}, actual={actual_tree}"
            )
        actual_sha256 = hashlib.sha256(patch).hexdigest()
        if actual_sha256 != expected_sha256.removeprefix("sha256:"):
            raise RuntimeError(
                f"交付物 SHA256 不一致：expected={expected_sha256}, actual={actual_sha256}"
            )

        if evaluation_baseline is not None:
            if oci is None or runtime_adapter is not None:
                raise ValueError('Evaluation baseline requires OCI without runtime adapter')
            from evaluation_baseline import validate_hashes
            validate_hashes(evaluation_baseline)
            manifest_path = EVALUATION_BASELINE_PATH
            manifest = json.loads(manifest_path.read_text())
            if {name: entry['sha256'] for name, entry in manifest['files'].items()} != evaluation_baseline:
                raise ValueError('Evaluation baseline differs from trusted task hashes')
            applied = subprocess.run(
                ['sudo', '-n', sys.executable, '/logs/verifier/evaluation_baseline.py', str(manifest_path), str(self.repo)],
                check=True, capture_output=True, text=True, timeout=60)
            receipt = json.loads(applied.stdout)
            if {name: entry['sha256'] for name, entry in receipt['files'].items()} != evaluation_baseline:
                raise ValueError('Evaluation baseline installation hash mismatch')
            WORKDIR.mkdir(parents=True, exist_ok=True)
            (WORKDIR / f'evaluation-baseline-rank-{self.index}.json').write_text(json.dumps(receipt, indent=2) + '\n')

        if oci is not None:
            self._apply_oci_layer(patch, actual_sha256, oci)
        elif patch:
            patch_path = Path("/tmp/harbor-engine.patch")
            patch_path.write_bytes(patch)
            subprocess.run(
                [
                    "sudo",
                    "-n",
                    "git",
                    "-C",
                    str(self.repo),
                    "apply",
                    "--check",
                    str(patch_path),
                ],
                check=True,
            )
            subprocess.run(
                ["sudo", "-n", "git", "-C", str(self.repo), "apply", str(patch_path)],
                check=True,
            )

        if not self.launch_script.is_file():
            raise FileNotFoundError(f"patch 未提供启动脚本：{self.launch_script}")
        subprocess.run(
            ["sudo", "-n", "chmod", "+x", str(self.launch_script)], check=True
        )
        # agent 阶段以 root 开发,repo 内可能留下 root 属主目录;launch 以当前
        # 用户执行(root+sudo -E 会因 HOME/PYTHONPATH 进入 root 上下文触发
        # torch TORCH_LIBRARY 双注册,regrade1 实测 import 即崩),故整树 chown
        # 移除写权限障碍(r5 实测 _cann_ops_custom/vendors Permission denied)。
        subprocess.run(
            ["sudo", "-n", "chown", "-R",
             f"{os.getuid()}:{os.getgid()}", str(self.repo)],
            check=True,
        )

        if runtime_adapter is not None:
            self._apply_runtime_adapter(runtime_adapter, actual_sha256, expected_patch_tree)
        return self._launch(head_ip)

    def _apply_oci_layer(self, layer: bytes, sha256: str, oci: dict[str, object]) -> None:
        """以 root 把 OCI layer 叠到本 pod 的 /,并核对应用器回报的 digest 与元数据一致。"""
        layer_path = OCI_LAYER_TMP
        layer_path.write_bytes(layer)
        script = OCI_LAYER_SCRIPT
        if "helper_source" in oci:
            # Ray actor 可能落在 worker 节点:那里没有宿主机上传到 /logs/verifier 的文件,
            # 应用器源码与层字节一样经 Ray 传入,本地落盘后执行(kimi-k3 0908 实测)。
            WORKDIR.mkdir(parents=True, exist_ok=True)
            local = WORKDIR / "oci_layer.py"
            local.write_text(str(oci["helper_source"]))
            script = str(local)
        forbidden = ",".join(str(p) for p in oci.get("forbidden_paths", []))
        output = subprocess.check_output(
            ["sudo", "-n", sys.executable, script, str(layer_path), "/", forbidden],
            text=True,
        )
        summary = json.loads(output)
        if summary["layer_sha256"] != f"sha256:{sha256}":
            raise RuntimeError(f"OCI layer 应用器回报的 layer_sha256 与交付物不一致：{summary}")
        if summary["diff_id"] != oci["diff_id"]:
            raise RuntimeError(
                f"OCI layer diff_id 不一致：expected={oci['diff_id']}, actual={summary['diff_id']}"
            )
        print(json.dumps({"oci_layer": summary}, ensure_ascii=False), flush=True)
        # 证据落盘:verifier 收尾时下载到 trial 的 verifier/ 目录,oracle CI 据此核对
        # "干净 serve job 叠的正是 collect 收到的那一层"。
        WORKDIR.mkdir(parents=True, exist_ok=True)
        (WORKDIR / f"oci-layer-rank-{self.index}.json").write_text(
            json.dumps({"rank": self.index, **summary}, ensure_ascii=False, indent=2) + "\n"
        )

    def _apply_runtime_adapter(self, adapter: tuple[Path, str], patch_sha: str, expected_tree: str) -> None:
        """patch 已应用并校验后执行可信 adapter；记录真实 pre/post 树和文件，不冒充原 patch 树。"""
        path, expected_sha = adapter
        path = path.resolve(strict=True)
        if not path.is_file() or path.is_relative_to(self.repo.resolve()):
            raise ValueError("runtime adapter 必须位于已校验引擎树之外")
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError("runtime adapter SHA256 不一致")
        before_tree = worktree_tree(self.repo, self.index)
        if expected_tree and before_tree != expected_tree:
            raise ValueError("adapter 执行前 candidate final_tree 不一致")
        execution = subprocess.run([sys.executable, "-I", str(path), str(self.repo)],
                                   cwd="/tmp", check=True, capture_output=True, text=True, timeout=60)
        result = json.loads(execution.stdout)
        if not isinstance(result, dict) or result.get("schema") != "runtime-adapter-result/1":
            raise ValueError("runtime adapter 输出协议不识别")
        files = result.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("runtime adapter 必须报告实际文件 pre/post SHA")
        for item in files:
            relative = Path(item["path"])
            target = (self.repo / relative).resolve(strict=True)
            if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(self.repo.resolve()):
                raise ValueError("runtime adapter 文件证据越界")
            if (len(item["pre_sha256"]) != 64 or
                    hashlib.sha256(target.read_bytes()).hexdigest() != item["post_sha256"]):
                raise ValueError("runtime adapter 文件 SHA 证据不一致")
        after_tree = worktree_tree(self.repo, self.index)
        evidence = {"schema": "runtime-adapter-evidence/1", "rank": self.index,
                    "host": socket.gethostname(), "adapter_sha256": actual_sha,
                    "patch_sha256": patch_sha, "engine_tree_pre_adapter": before_tree,
                    "engine_tree_post_adapter": after_tree,
                    "runtime_differs_from_patch_tree": before_tree != after_tree,
                    "adapter_result": result}
        WORKDIR.mkdir(parents=True, exist_ok=True)
        (WORKDIR / f"runtime-adapter-rank-{self.index}.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        self.runtime_adapter_evidence = evidence

    def _launch(self, head_ip: str) -> dict[str, object]:
        """启动 launch.sh 为独立进程组并返回初始状态。"""
        if not self.launch_script.is_file():
            raise FileNotFoundError(f"缺少启动脚本：{self.launch_script}")
        WORKDIR.mkdir(parents=True, exist_ok=True)
        self.rc_path.unlink(missing_ok=True)
        self.log_file = self.log_path.open("wb")
        env = os.environ.copy()
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "ASCEND_RT_VISIBLE_DEVICES",
            "NPU_VISIBLE_DEVICES",
            # 平台/driver 的全局默认值不能强迫独立副本争用同一个 TCPStore。
            # task 如需固定端口，必须在 launch.sh 内按进程组显式设置。
            "VLLM_PORT",
            "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE",
        ):
            env.pop(name, None)
        if head_ip:
            env["RAY_IP"] = head_ip
        # launch.sh 靠这两个决定本节点是起 Ray head 还是 join(逐 pod 路径)。
        env["HARBOR_SERVE_RANK"] = str(self.index)
        if os.environ.get("HARBOR_SERVE_HEAD_IPV4"):
            env["HARBOR_SERVE_HEAD_IPV4"] = os.environ["HARBOR_SERVE_HEAD_IPV4"]
        # 外层 bash 在 launch.sh(含 exec 的服务进程)退出后写 rc 文件;
        # 服务健康时 exec 的 vllm/sglang 常驻,rc 永不出现。
        self.process = subprocess.Popen(
            ["bash", "-c", launch_wrapper_command(self.launch_script, self.rc_path)],
            cwd=self.repo,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.pid_path.write_text(f"{self.process.pid}\n")
        time.sleep(2)
        result = self.status()
        if self.runtime_adapter_evidence is not None:
            result["runtime_adapter"] = self.runtime_adapter_evidence
        return result

    def status(self) -> dict[str, object]:
        """返回当前节点的服务进程状态。"""
        if self.log_file is not None:
            self.log_file.flush()
        return_code = None if self.process is None else self.process.poll()
        return {
            "host": socket.gethostname(),
            "rank": self.index,
            "pid": None if self.process is None else self.process.pid,
            "running": self.process is not None and return_code is None,
            "return_code": return_code,
            "log": str(self.log_path),
        }

    def diagnostics(self) -> dict[str, object]:
        """Ray actor 保有固定 rank，读取真实 wrapper rc 和有界日志，不重新部署。"""
        value = self.status()
        if self.rc_path.exists():
            value["launch_rc"] = int(self.rc_path.read_text().strip())
        else:
            value["launch_rc"] = None
        value["tail"] = ""
        if self.log_path.exists():
            with self.log_path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 3000))
                value["tail"] = stream.read().decode(errors="replace")
        return value

    def stop(self) -> dict[str, object]:
        """停止本节点服务进程组。"""
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=30)
        self.pid_path.unlink(missing_ok=True)
        return self.status()


def submission_checks(metadata: dict[str, object], forbidden_paths: list[str]) -> dict[str, object]:
    """从 collect 元数据得出 apply_and_start 的校验参数:OCI layer(layer_sha256)为默认协议,
    patch_sha256 为 legacy;缺基线字段直接报错。"""
    missing = [key for key in ("base_commit", "baseline_tree") if key not in metadata]
    if "layer_sha256" not in metadata and "patch_sha256" not in metadata:
        missing.append("layer_sha256|patch_sha256")
    if missing:
        raise ValueError(f"交付物 metadata 缺少字段：{missing}")
    if "layer_sha256" in metadata:
        if "diff_id" not in metadata:
            raise ValueError("OCI layer metadata 缺少 diff_id")
        return {
            "expected_sha256": str(metadata["layer_sha256"]),
            "oci": {"diff_id": str(metadata["diff_id"]), "forbidden_paths": list(forbidden_paths)},
        }
    return {"expected_sha256": str(metadata["patch_sha256"]), "oci": None}


def run_node(
    patch: bytes,
    metadata: dict[str, object],
    engine_repo: str,
    rank: int = 0,
    head_ip_override: str = "",
    head_ipv4: str = "",
    relaunch: bool = False,
    oracle_script: str = "",
    runtime_adapter: tuple[Path, str] | None = None,
    forbidden_paths: list[str] | None = None,
    evaluation_baseline: dict[str, str] | None = None,
) -> None:
    """在**本 pod** 上应用交付物并启动服务，不依赖平台预起的 Ray 集群。

    多节点训练题由宿主机侧逐 pod 调用本脚本(每次带自己的 --rank 与
    --head-ip)，而不是由某个 pod 用 Ray actor 去摆放其它节点。原因：平台的
    `enable_ray_on_arnold` 在**多节点**时会把 job 切到 Arnold 的
    robust-training entrypoint，那条路径**不配置 Merlin 的 sshd**，于是
    Harbor 的 SSH exec 根本连不上(job 一直 RUNNING，只报
    `Exec channel did not become ready`)。既然不能依赖平台起集群，跨节点的
    Ray 就交给引擎自己的 launch.sh 起 —— rank 与 head 地址通过环境变量
    HARBOR_SERVE_RANK / RAY_IP 传进去，launch.sh 按 rank 决定
    `ray start --head` 还是 `ray start --address`。这样 serve_controller
    保持引擎无关，Ray 的版本与拉起方式全部留在 task 自己的 launch.sh 里
    (也顺带消掉了"系统 system Ray 与 venv 开源 ray 两套并存"的隐患)。
    """
    checks: dict[str, object] = {"expected_sha256": "", "oci": None}
    if not oracle_script and not relaunch:
        checks = submission_checks(metadata, forbidden_paths or [])
    runner = ServiceRunner(rank, engine_repo)
    head_ip = (
        head_ip_override
        or os.environ.get("RAY_IP")
        or os.environ.get("BYTED_RAY_POD_IP")
        or ""
    )
    if head_ipv4:
        os.environ["HARBOR_SERVE_HEAD_IPV4"] = head_ipv4
    result = runner.apply_and_start(
        patch,
        str(checks["expected_sha256"]),
        str(metadata.get("base_commit", "")),
        str(metadata.get("baseline_tree", "")),
        head_ip,
        relaunch=relaunch,
        oracle_script=oracle_script,
        oci=checks["oci"],
        **({'evaluation_baseline': evaluation_baseline} if evaluation_baseline is not None else {}),
        **({"runtime_adapter": runtime_adapter, "expected_patch_tree": str(metadata.get("final_tree", ""))}
           if runtime_adapter is not None else {}),
    )
    print(json.dumps([result], ensure_ascii=False), flush=True)
    if not result["running"]:
        raise RuntimeError(f"服务进程启动失败：{result}")


def run_single_node(
    patch: bytes, metadata: dict[str, object], engine_repo: str,
    relaunch: bool = False, oracle_script: str = "",
    forbidden_paths: list[str] | None = None,
) -> None:
    """单节点(rank 0)入口;逐 pod 显式 rank 的路径直接走 run_node。"""
    run_node(
        patch, metadata, engine_repo,
        relaunch=relaunch, oracle_script=oracle_script, forbidden_paths=forbidden_paths,
    )


def run_multi_node(
    patch: bytes,
    metadata: dict[str, object],
    engine_repo: str,
    node_count: int,
    gpus_per_node: int,
    oracle_script: str = "",
    forbidden_paths: list[str] | None = None,
) -> None:
    """多节点(平台预起 Ray 的推理题):等待全部 Ray 节点就绪,创建固定节点
    actor 并并行启动。训练题的多节点不走这条 —— 见 run_node 的逐 pod 路径。"""
    checks: dict[str, object] = {"expected_sha256": "", "oci": None}
    if not oracle_script:
        checks = submission_checks(metadata, forbidden_paths or [])
        if checks["oci"] is not None:
            # worker 节点没有 /logs/verifier 上传件:应用器源码随 actor 参数一起下发。
            checks["oci"]["helper_source"] = Path(OCI_LAYER_SCRIPT).read_text()
    head_service_port = os.environ.get("HARBOR_SERVICE_PORT")
    if not head_service_port:
        raise RuntimeError("HARBOR_SERVICE_PORT 必须由 head 注入，不能猜测多节点服务端口")
    # 与 HTTP 一样,内部口必须来自 head;worker 本地 PORT1/2 不能当 dist-init。
    head_port_env = {
        "HARBOR_SERVICE_PORT": head_service_port,
    }
    for key in (
        "HARBOR_SERVICE_HOST",
        "HARBOR_DIST_PORT",
        "HARBOR_DP_RPC_PORT",
        "VLLM_PORT",
        "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE",
    ):
        value = os.environ.get(key)
        if value:
            head_port_env[key] = value
    if "HARBOR_DIST_PORT" not in head_port_env:
        raise RuntimeError("HARBOR_DIST_PORT 必须由 head 注入，不能回退 bind(0) 或写死 5000")
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    remote_runner = ray.remote(num_cpus=0, max_restarts=0)(ServiceRunner)
    ray.init(address=wait_for_ray_gcs(), namespace=NAMESPACE, logging_level="ERROR")
    nodes = []
    for attempt in range(1, 91):
        nodes = [node for node in ray.nodes() if node["Alive"]]
        if len(nodes) == node_count:
            break
        print(
            f"等待 {node_count} 个 Ray 节点注册：当前 {len(nodes)}/{node_count}"
            f"（第 {attempt}/90 次）",
            flush=True,
        )
        time.sleep(10)
    if len(nodes) != node_count:
        raise RuntimeError(
            f"15 分钟内未等到 {node_count} 个存活 Ray 节点，实际为 {len(nodes)}"
        )

    nodes.sort(
        key=lambda node: (
            0 if node["Resources"].get("head") else 1,
            node["NodeManagerAddress"],
        )
    )
    head_ip = os.environ.get("RAY_IP") or os.environ.get("BYTED_RAY_POD_IP")
    if not head_ip:
        raise RuntimeError("RAY_IP/BYTED_RAY_POD_IP 必须存在")
    actors = []
    for index, node in enumerate(nodes):
        actor = remote_runner.options(
            name=f"{ACTOR_PREFIX}{index}",
            num_gpus=gpus_per_node,
            lifetime="detached",
            # Ray worker 不继承 driver 的 shell env；所有 rank 使用同一可信 head 端口。
            runtime_env={"env_vars": head_port_env},
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"], soft=False
            ),
        ).remote(index, engine_repo)
        actors.append(actor)

    results = ray.get(
        [
            actor.apply_and_start.remote(
                patch,
                # oracle 模式没有 metadata(三项校验被跳过),用 get 取占位值;
                # 交付物模式仍要求字段齐全,缺失时 submission_checks 给出明确报错。
                checks["expected_sha256"],
                metadata.get("base_commit", ""),
                metadata.get("baseline_tree", ""),
                head_ip,
                oracle_script=oracle_script,
                oci=checks["oci"],
            )
            for actor in actors
        ]
    )
    print(json.dumps(results, ensure_ascii=False), flush=True)
    failed = [result for result in results if not result["running"]]
    if failed:
        raise RuntimeError(f"服务进程启动失败：{failed}")


def main() -> None:
    """应用 patch 并启动服务。

    显式 `--rank` = 宿主机逐 pod 调用(训练题多节点);缺省则按 config 拓扑
    选择单机直启或 Ray 多节点路径(推理题)。两条 CLI 姿势必须同时成立:
    infer_verifier 用旧姿势(位置参数 + --relaunch/--oracle),patch_eval /
    train_verifier 用新姿势(--rank/--head-ip/--head-ipv4)。
    """
    parser = argparse.ArgumentParser(description="应用交付物(OCI layer / legacy patch)并启动服务")
    parser.add_argument(
        "paths",
        nargs="+",
        help="交付物(submit_oci.tar.gz 或 legacy patch) metadata config 三个路径;--oracle 模式只需 config(取最后一个)",
    )
    parser.add_argument(
        "--relaunch", action="store_true",
        help="部署重试:跳过校验,清掉残余引擎进程后直接重启 launch.sh",
    )
    parser.add_argument(
        "--oracle", default="",
        help="oracle 基线模式:先在引擎树上执行该脚本(产出 launch.sh)再启动",
    )
    parser.add_argument(
        "--rank", type=int, default=None,
        help="本节点在服务拓扑中的序号(宿主机逐 pod 调用时显式传入),head 为 0",
    )
    parser.add_argument("--head-ip", default="", help="rank 0 的对外地址(全局 IPv6)")
    parser.add_argument(
        "--head-ipv4",
        default="",
        help="rank 0 的 IPv4，供引擎内部组网(有的引擎的 host:port 拼接不支持 IPv6)",
    )
    parser.add_argument("--runtime-adapter", default="", help="可信调用方上传的 adapter，仅支持显式 --rank")
    parser.add_argument("--runtime-adapter-sha256", default="", help="可信宿主计算的 adapter SHA256")
    args = parser.parse_args()
    runtime_adapter = None
    if args.runtime_adapter or args.runtime_adapter_sha256:
        if (not args.runtime_adapter or args.rank is None or len(args.runtime_adapter_sha256) != 64
                or any(c not in "0123456789abcdef" for c in args.runtime_adapter_sha256)):
            raise ValueError("runtime adapter 必须有显式 rank、路径和可信 SHA256")
        runtime_adapter = (Path(args.runtime_adapter), args.runtime_adapter_sha256)

    # oracle 模式不读 patch/metadata:三项校验只对 agent 交付物有意义。
    if args.oracle:
        config_path = Path(args.paths[-1])
        patch, metadata = b"", {}
    else:
        if len(args.paths) != 3:
            raise SystemExit("交付物模式需要 <layer|patch> <metadata> <config> 三个路径")
        patch = Path(args.paths[0]).read_bytes()
        metadata = json.loads(Path(args.paths[1]).read_text())
        config_path = Path(args.paths[2])

    # launch_env 由 task 的 verifier-config 声明,注入 launch.sh 的环境
    # (训练题用;推理题 config 不写该字段即 no-op)。
    config = json.loads(config_path.read_text())
    launch_env = config.get("launch_env", {})
    if not isinstance(launch_env, dict):
        raise SystemExit("launch_env 必须是 JSON object")
    for name, value in launch_env.items():
        if name in {"HARBOR_SERVICE_PORT", "HARBOR_DIST_PORT", "RAY_PORT"}:
            raise SystemExit(f"launch_env 不得覆盖可信动态端口: {name}")
        if not isinstance(name, str) or not name or "=" in name:
            raise SystemExit(f"非法 launch_env 名称：{name!r}")
        if not isinstance(value, (str, int, float, bool)):
            raise SystemExit(f"launch_env.{name} 必须是标量")
        os.environ[name] = str(value)
    engine_repo, node_count, gpus_per_node = load_serve_config(config_path)
    if config.get('evaluation_baseline') is not None and (args.rank is None or args.oracle or args.relaunch):
        raise ValueError('Evaluation baseline requires explicit rank and full OCI verification')
    # task 追加的禁止路径(只能追加,公共集合在 oci_layer.FORBIDDEN 里不可放宽)。
    forbidden_paths = config.get("forbidden_paths", [])
    if not isinstance(forbidden_paths, list) or not all(isinstance(p, str) and p for p in forbidden_paths):
        raise SystemExit("forbidden_paths 必须是非空字符串列表")

    if args.rank is not None:
        if not 0 <= args.rank < node_count:
            raise SystemExit(
                f"--rank {args.rank} 超出 serve_node_count={node_count} 的范围"
            )
        run_node(
            patch, metadata, engine_repo, args.rank, args.head_ip, args.head_ipv4,
            relaunch=args.relaunch, oracle_script=args.oracle, forbidden_paths=forbidden_paths,
            **({'evaluation_baseline': config['evaluation_baseline']} if config.get('evaluation_baseline') is not None else {}),
            **({"runtime_adapter": runtime_adapter} if runtime_adapter is not None else {}),
        )
    elif node_count == 1:
        run_single_node(
            patch, metadata, engine_repo, relaunch=args.relaunch,
            oracle_script=args.oracle, forbidden_paths=forbidden_paths,
        )
    else:
        if args.relaunch:
            raise ValueError("--relaunch 仅支持单节点拓扑")
        run_multi_node(
            patch, metadata, engine_repo, node_count, gpus_per_node,
            oracle_script=args.oracle, forbidden_paths=forbidden_paths,
        )


if __name__ == "__main__":
    main()
