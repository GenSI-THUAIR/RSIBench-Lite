"""task.yaml → task.toml + tests/verifier-config.json 的编译器。

开发期用法(只处理含 task.yaml 的目录):

    python -m schemas.build tasks/infer-qwen3p8-27b          # 生成/覆盖
    python -m schemas.build --check tasks/infer-qwen3p8-27b  # 只比对不落盘
    python -m schemas.build tasks/*/                        # 全量
    python -m schemas.build --clean tasks/*/                # 删除生成物

运行入口通过 schemas.run_config 在每次独占的快照中编译，不写回源任务。
以下 CLI 用于开发期检查；--clean 只清理带 task.yaml 的目录，手写
task.toml 的题不受影响。

资源分配由 Modal 运行配置编译器处理；此模块保留任务及评分协议的严格加载与发射。
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

from schemas.performance_baseline import validate_performance_baseline

# ---------------------------------------------------------------------------
# 严格 YAML 加载:补回 TOML 天然具备的响亮失败
# ---------------------------------------------------------------------------


class StrictYamlError(ValueError):
    """task.yaml 违反严格加载规则。"""


class _StrictLoader(yaml.SafeLoader):
    def compose_node(self, parent: Any, index: Any) -> Any:
        event = self.peek_event()
        if isinstance(event, yaml.events.AliasEvent):
            raise StrictYamlError(
                f"YAML alias '*{event.anchor}' is forbidden "
                f"(line {event.start_mark.line + 1})."
            )
        if getattr(event, "anchor", None):
            raise StrictYamlError(
                f"YAML anchor '&{event.anchor}' is forbidden "
                f"(line {event.start_mark.line + 1})."
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
        seen: set[str] = set()
        for key_node, _value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise StrictYamlError(
                    f"YAML merge key '<<' is forbidden "
                    f"(line {key_node.start_mark.line + 1})."
                )
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, str):
                raise StrictYamlError(
                    f"Non-string mapping key {key!r} "
                    f"(line {key_node.start_mark.line + 1})."
                )
            if key in seen:
                raise StrictYamlError(
                    f"Duplicate key {key!r} (line {key_node.start_mark.line + 1})."
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


class _Ref:
    """``!ref dotted.path`` 的载荷:同文件内值引用,load 后由 _resolve_refs 消解。

    与 harbor 运行时的 ``${VAR}`` env 插值刻意不同源:!ref 是 YAML tag,在
    build 侧解析完就不存在了;${VAR} 原样透传给 harbor 在宿主机解析。
    """

    __slots__ = ("path",)

    def __init__(self, path: str) -> None:
        self.path = path


_StrictLoader.add_constructor(
    "!ref", lambda loader, node: _Ref(loader.construct_scalar(node))
)


# 收窄 YAML 1.1 隐式类型:bool 只认 true/false(yes/no/on/off 保持字符串),
# 裸日期保持字符串。写错类型由 TOML 发射时的类型分派暴露,而不是静默变形。
_DROPPED_TAGS = frozenset(
    {"tag:yaml.org,2002:bool", "tag:yaml.org,2002:timestamp", "tag:yaml.org,2002:value"}
)
_StrictLoader.yaml_implicit_resolvers = {
    prefix: [(tag, regexp) for tag, regexp in resolvers if tag not in _DROPPED_TAGS]
    for prefix, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$"), list("tf")
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        data = yaml.load(path.read_text(), Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise StrictYamlError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise StrictYamlError(
            f"{path}: top level must be a mapping, got {type(data).__name__}."
        )
    return _resolve_refs(data)


def _resolve_refs(root: dict[str, Any]) -> dict[str, Any]:
    """消解 ``!ref``:按点路径(支持数字段做列表索引)取根下的值,深拷贝
    落位(引用双方从此互不共享可变对象);未知路径与循环引用报错。"""
    resolving: set[str] = set()

    def lookup(path: str) -> Any:
        node: Any = root
        for part in path.split("."):
            if isinstance(node, _Ref):
                node = resolve(node)
            if isinstance(node, list):
                node = node[int(part)]
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                raise StrictYamlError(f"!ref 未知路径: {path}")
        return node

    def resolve(value: Any) -> Any:
        if isinstance(value, _Ref):
            if value.path in resolving:
                raise StrictYamlError(f"!ref 循环引用: {value.path}")
            resolving.add(value.path)
            try:
                return copy.deepcopy(resolve(lookup(value.path)))
            finally:
                resolving.discard(value.path)
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    return resolve(root)


# ---------------------------------------------------------------------------
# TOML 值的序列化(只覆盖 task.toml 实际用到的形状)
# ---------------------------------------------------------------------------


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        if "\n" in value:
            return _toml_multiline(value)
        # TOML 基本字符串的转义集覆盖 JSON 输出;借用 json 的引号与转义。
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        entries = ", ".join(
            f"{key} = {_toml_value(item)}" for key, item in value.items()
        )
        return "{ " + entries + " }"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _toml_multiline(value: str) -> str:
    """多行字符串走三引号字面量,沿用现有 task.toml 的书写惯例。"""
    if "'''" in value or not value.endswith("\n"):
        raise ValueError(
            "Multiline string cannot be emitted as a TOML literal (contains "
            "''' or lacks trailing newline); adjust the YAML content."
        )
    return "'''\n" + value + "'''"


def emit_task_toml(cfg: dict[str, Any]) -> str:
    lines: list[str] = []

    def kv(key: str, value: Any) -> None:
        lines.append(f"{key} = {_toml_value(value)}")

    def section(header: str, table: dict[str, Any], skip: tuple[str, ...] = ()) -> None:
        lines.append("")
        lines.append(f"[{header}]")
        for key, value in table.items():
            if key not in skip:
                kv(key, value)

    def env_section(header: str, table: dict[str, Any]) -> None:
        """Emit explicitly configured runtime environment variables."""
        env = table.get("env", {})
        if env:
            section(f"{header}.env", env)

    kv("schema_version", cfg["schema_version"])
    if "artifacts" in cfg:
        lines.append("artifacts = [")
        for artifact in cfg["artifacts"]:
            lines.append(f"  {_toml_value(artifact)},")
        lines.append("]")

    section("task", cfg["task"])
    if "metadata" in cfg:
        section("metadata", cfg["metadata"])
    environment = cfg["environment"]
    section(
        "environment",
        environment,
        skip=("env", "mounts", "nodes"),
    )
    env_section("environment", environment)
    if "agent" in cfg:
        section("agent", cfg["agent"])

    verifier = cfg.get("verifier", {})
    if verifier:
        section("verifier", verifier, skip=("environment", "collect", "scoring"))
        verifier_env = verifier.get("environment")
        if verifier_env is not None:
            section(
                "verifier.environment",
                verifier_env,
                skip=("env", "mounts", "nodes"),
            )
            env_section("verifier.environment", verifier_env)
        for collect in verifier.get("collect", ()):
            lines.append("")
            lines.append("[[verifier.collect]]")
            for key, value in collect.items():
                kv(key, value)

    return "\n".join(lines) + "\n"


def emit_verifier_config(cfg: dict[str, Any]) -> str | None:
    verifier_config = cfg.get("verifier", {}).get("scoring")
    if verifier_config is None:
        return None
    return json.dumps(verifier_config, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# 构建与比对
# ---------------------------------------------------------------------------


def _diff_paths(old: Any, new: Any, prefix: str = "") -> list[str]:
    """定位两棵解析树的差异路径,给出人能读的报告。"""
    if type(old) is not type(new):
        return [f"{prefix or '<root>'}: {type(old).__name__} → {type(new).__name__}"]
    if isinstance(old, dict):
        out: list[str] = []
        for key in old.keys() | new.keys():
            where = f"{prefix}.{key}" if prefix else key
            if key not in new:
                out.append(f"{where}: 仅旧文件有")
            elif key not in old:
                out.append(f"{where}: 仅生成物有")
            else:
                out.extend(_diff_paths(old[key], new[key], where))
        return out
    if isinstance(old, list):
        if len(old) != len(new):
            return [f"{prefix}: 长度 {len(old)} → {len(new)}"]
        out = []
        for index, (o, n) in enumerate(zip(old, new)):
            out.extend(_diff_paths(o, n, f"{prefix}[{index}]"))
        return out
    if old != new:
        return [f"{prefix}: {old!r} → {new!r}"]
    return []


# Collector 是交付协议脚本；这里只读取并嵌入，不加入 Benchmark payload。
_OCI_COLLECTOR = Path(__file__).resolve().parents[1] / "verify" / "delivery" / "collect_oci_layer.py"


def _oci_collect_command(scoring: dict[str, Any]) -> str:
    """把共享 OCI collector 渲染成 Agent Job 内执行的 collect 命令(heredoc 嵌入)。"""
    script = _OCI_COLLECTOR.read_text()
    if "\nPY\n" in script:
        raise ValueError(f"{_OCI_COLLECTOR} 含 heredoc 结束标记 PY")
    return (
        "set -euo pipefail\n"
        f"REPO=\"${{HARBOR_ENGINE_REPO:-{scoring['engine_repo']}}}\"\n"
        "OUT=\"${HARBOR_PATCH_OUTPUT_DIR:-/logs/submission}\"\n"
        f"PYTHON_BIN=\"${{HARBOR_PATCH_PYTHON:-{scoring['python']}}}\"\n"
        "BASELINE_TREE_FILE=\"${HARBOR_BASELINE_TREE_FILE:-/tmp/.harbor-git-baseline-tree}\"\n"
        f"\"$PYTHON_BIN\" - \"$REPO\" \"$OUT\" \"$BASELINE_TREE_FILE\" {scoring['layer_artifact']} <<'PY'\n"
        f"{script}"
        "PY\n"
    )

def _materialize(cfg: dict[str, Any]) -> dict[str, Any]:
    """Derive delivery artifacts and service topology without platform job requests."""
    if "prose" in cfg:
        raise ValueError("prose is no longer supported")
    wire = copy.deepcopy(cfg)

    def derive(env):
        return int(env.pop("nodes", 1)), int(env["gpus"])

    # artifacts 由交付物名派生:layer_artifact(OCI layer,默认协议)或
    # patch_artifact(legacy Git patch)二选一,与 patch-metadata.json 恒成对。
    scoring = wire.get("verifier", {}).get("scoring", {})
    layer = scoring.get("layer_artifact")
    patch = scoring.get("patch_artifact")
    if layer and patch:
        raise ValueError("verifier.scoring 的 layer_artifact 与 patch_artifact 只能二选一")
    artifact = layer or patch
    if artifact and "artifacts" not in wire:
        wire["artifacts"] = [
            {"source": f"/logs/submission/{artifact}", "destination": artifact},
            {
                "source": "/logs/submission/patch-metadata.json",
                "destination": "patch-metadata.json",
            },
        ]
    # layer 型 task 的 collect hook 由共享 collector 派生,不在每题内联。
    if layer and "collect" not in wire["verifier"]:
        wire["verifier"]["collect"] = [
            {"timeout_sec": 300.0, "user": "root", "command": _oci_collect_command(scoring)}
        ]

    nodes, gpus = derive(wire["environment"])
    verifier_env = wire.get("verifier", {}).get("environment")
    if verifier_env is not None:
        derive(verifier_env)

    vc = wire.get("verifier", {}).get("scoring")
    if vc is not None and "serve_node_count" not in vc:
        ordered: dict[str, Any] = {}
        for key, value in vc.items():
            ordered[key] = value
            if key == "service_label":
                ordered["serve_node_count"] = nodes
                ordered["serve_gpus_per_node"] = gpus
        wire["verifier"]["scoring"] = ordered
    return wire


def build_task(task_dir: Path, *, check: bool) -> list[str]:
    """构建一个 task 目录;返回问题列表(空即成功)。"""
    cfg = load_yaml(task_dir / "task.yaml")
    scoring = cfg.get("verifier", {}).get("scoring")
    if scoring is not None:
        validate_performance_baseline(scoring)
    wire = _materialize(cfg)
    problems: list[str] = []

    outputs: list[tuple[Path, str, str]] = []  # (路径, 内容, 解析格式)
    outputs.append((task_dir / "task.toml", emit_task_toml(wire), "toml"))
    verifier_config_text = emit_verifier_config(wire)
    if verifier_config_text is not None:
        outputs.append(
            (task_dir / "tests" / "verifier-config.json", verifier_config_text, "json")
        )
    # 题面与 task.toml / verifier-config.json 同生命周期(I3 已切换)。
    from schemas.instruction_template import render_from_config as render_instruction

    instruction_text = render_instruction(cfg)
    if instruction_text is not None:
        outputs.append((task_dir / "instruction.md", instruction_text, "md"))

    for path, text, fmt in outputs:
        if fmt == "md":
            # 题面是自由文本,无解析式比对;写模式直接落盘,check 仅验可渲染。
            if not check:
                path.write_text(text)
                print(f"{path}: 已从 task.yaml 生成")
            continue
        parse = tomllib.loads if fmt == "toml" else json.loads
        # 生成物必须先能被解析回等价结构,防 emitter 自身出错。
        parsed = parse(text)
        drifts: list[str] = []
        if path.exists():
            existing = parse(path.read_text())
            drifts = [f"{path}: 语义漂移 {d}" for d in _diff_paths(existing, parsed)]
        if check:
            problems.extend(drifts)
            if not path.exists():
                problems.append(f"{path}: 不存在,--check 无从比对")
            elif not drifts and path.read_text() != text:
                print(f"# {path}: 语义一致,字节不同(注释/格式),写模式将归一")
        else:
            # 写模式:语义漂移意味着 YAML 是新的真值来源,如实覆盖。
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"{path}: 已从 task.yaml 生成")
    return problems


GENERATED_FILES = ("task.toml", "tests/verifier-config.json", "instruction.md")


def clean_task(task_dir: Path) -> None:
    """删除一个 task 目录的编译产物(仅当它以 task.yaml 为源)。"""
    for rel in GENERATED_FILES:
        path = task_dir / rel
        if path.exists():
            path.unlink()
            print(f"{path}: 已清除(编译产物,源在 task.yaml)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m schemas.build")
    parser.add_argument("task_dirs", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="只比对不落盘")
    parser.add_argument("--clean", action="store_true", help="删除生成物")
    args = parser.parse_args(argv)

    status = 0
    for task_dir in args.task_dirs:
        if not (task_dir / "task.yaml").is_file():
            continue
        if args.clean:
            clean_task(task_dir)
            continue
        problems = build_task(task_dir, check=args.check)
        for problem in problems:
            print(problem, file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
