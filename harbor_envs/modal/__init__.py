"""Modal 平台入口；纯配置模块不加载可选的 Modal SDK。

Harbor 配置继续使用 harbor_envs.modal:ModalEnvironment。
"""

__all__ = ["ModalEnvironment"]


def __getattr__(name: str):
    if name == "ModalEnvironment":
        from harbor_envs.modal.environment import ModalEnvironment
        return ModalEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
