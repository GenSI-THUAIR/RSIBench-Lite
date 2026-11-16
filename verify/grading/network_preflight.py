"""Conservative distributed-network preflight for verifier jobs.

Missing optional NCCL/HCCL variables are valid.  The preflight only fails for
an explicitly required variable, an explicit candidate/baseline mismatch, or
an interface that the target pod proves is absent.  Every decision is
serializable so a later failure can be attributed to configuration or
platform evidence rather than guessed from a timeout.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shlex
from typing import Any

from verify.delivery.patch_eval import ServiceEndpointError


NETWORK_VARIABLES = (
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "HCCL_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "HCCL_IF_IP",
)


class NetworkPreflightError(ServiceEndpointError):
    """The explicit network contract cannot be proven."""


def _interfaces(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def compare_network_env(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object] | None = None,
    *,
    required: Sequence[str] = (),
) -> dict[str, Any]:
    """Compare only declared network settings and preserve optional absence."""
    baseline = baseline or {}
    required_set = set(required)
    variables = sorted(set(NETWORK_VARIABLES) | required_set | {
        key for key in candidate if key.endswith("_IFNAME") or key == "HCCL_IF_IP"
    })
    missing_required = [key for key in sorted(required_set) if not candidate.get(key)]
    mismatches = [
        {"variable": key, "candidate": candidate.get(key), "baseline": baseline.get(key)}
        for key in variables
        if key in baseline and key in candidate and candidate[key] != baseline[key]
    ]
    return {
        "schema": "network-preflight/1",
        "status": "FAIL" if missing_required or mismatches else "PASS",
        "required": sorted(required_set),
        "missing_required": missing_required,
        "optional_missing": [key for key in variables if key not in required_set and not candidate.get(key)],
        "mismatches": mismatches,
        "variables": {key: {"candidate": candidate.get(key), "baseline": baseline.get(key)} for key in variables},
    }


async def probe_interfaces(environment: Any, interfaces: Sequence[str], pods: Sequence[str | None]) -> dict[str, Any]:
    """Check interface presence on each role pod using the provider exec path."""
    checked = []
    for pod in pods:
        for interface in interfaces:
            if not interface.replace("_", "").replace("-", "").isalnum():
                raise NetworkPreflightError(f"invalid interface name: {interface!r}")
            result = await environment.exec(
                f"test -d /sys/class/net/{shlex.quote(interface)}",
                cwd="/tmp", timeout_sec=30, pod_name=pod,
            )
            checked.append({
                "pod": pod,
                "interface": interface,
                "present": result.return_code == 0,
                "return_code": result.return_code,
                "stderr": (result.stderr or "")[-500:],
            })
    missing = [row for row in checked if not row["present"]]
    return {"checked": checked, "missing": missing, "status": "FAIL" if missing else "PASS"}


async def run_network_preflight(
    environment: Any,
    *,
    candidate_env: Mapping[str, object],
    baseline_env: Mapping[str, object] | None = None,
    required: Sequence[str] = (),
    pods: Sequence[str | None] = (None,),
    evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Run config comparison and interface probes; fail only explicit errors."""
    report = compare_network_env(candidate_env, baseline_env, required=required)
    if report["status"] == "FAIL":
        if evidence_path is not None:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise NetworkPreflightError(json.dumps(report, sort_keys=True))
    interfaces = sorted({name for key in NETWORK_VARIABLES if key.endswith("IFNAME")
                         for name in _interfaces(candidate_env.get(key))})
    if interfaces:
        report["interfaces"] = await probe_interfaces(environment, interfaces, pods)
        if report["interfaces"]["status"] == "FAIL":
            report["status"] = "FAIL"
    else:
        report["interfaces"] = {"checked": [], "missing": [], "status": "SKIPPED"}
    if evidence_path is not None:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] == "FAIL":
        raise NetworkPreflightError(json.dumps(report, sort_keys=True))
    return report
