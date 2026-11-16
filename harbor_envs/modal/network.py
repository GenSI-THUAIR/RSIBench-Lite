"""Modal 注入的服务网络与驱动约定；题目启动脚本不判断平台。

Modal 每个 Sandbox 独立网络空间，内部通信使用 IPv4 回环和固定端口。
"""

SERVICE_HOST_ENV = "HARBOR_SERVICE_HOST"


def modal_driver_env(engine: str) -> dict[str, str]:
    """gVisor使用宿主提供的CUDA适配，不能用vLLM的compat库覆盖它。"""
    return {"HARBOR_CUDA_COMPATIBILITY": "0"} if engine == "vllm" else {}


def configure_job_network(config: dict, environment: str | None = None) -> bool:
    """按最终选中的 Harbor env 补齐网络配置；动态平台端口留给各后端解析。"""
    selected = config.get("environment", {})
    backend = environment or selected.get("import_path") or selected.get("type")
    if backend not in {"modal", "harbor.environments.modal:ModalEnvironment",
                       "harbor_envs.modal:ModalEnvironment"}:
        return False
    env = config.setdefault("environment", {}).setdefault("env", {})
    env.update(modal_network_env())
    return True


def modal_network_env(service_port: int = 8000) -> dict[str, str]:
    """返回单服务 Sandbox 的完整网络环境，供后端统一传给全部执行阶段。"""
    if type(service_port) is not int or not 1024 <= service_port <= 65535:
        raise ValueError("service_port must be an integer between 1024 and 65535")
    if service_port in {29500, 29501}:
        raise ValueError("service_port overlaps distributed communication ports")
    return {
        SERVICE_HOST_ENV: "0.0.0.0",
        "HARBOR_SERVICE_PORT": str(service_port),
        "HARBOR_DIST_PORT": "29500",
        "HARBOR_DP_RPC_PORT": "29501",
        "VLLM_PORT": "29500",
        "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE": "tcp://127.0.0.1:29500",
    }
