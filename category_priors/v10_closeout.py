from __future__ import annotations

"""Close V9 without preserving its premature identity-limit conclusion."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import load_json, write_json


_STATUS_NAMES = (
    "v9_orchestrator_status.json",
    "v9_status.json",
    "v9_stage3_6_status.json",
)


def write_v10_v9_closeout(
    *,
    v9_artifacts_root: Path,
    output_path: Path,
    git_commit: str,
) -> dict[str, Any]:
    """Record the exact conclusion V10 is allowed to inherit from V9.

    ``passed`` means that the *correction* is closed and reproducible.  It does
    not mean V9 passed its ObjectBank gate.  The source statuses remain
    read-only and are referenced by path/state/checkpoint rather than copied or
    hashed.
    """

    commit = str(git_commit).strip()
    if not commit:
        raise ValueError("V10 closeout git_commit must be non-empty")
    root = Path(v9_artifacts_root).resolve()
    statuses: dict[str, dict[str, Any]] = {}
    for name in _STATUS_NAMES:
        path = root / name
        if not path.is_file():
            continue
        payload = load_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"V9 status is not a JSON object: {path}")
        statuses[name] = {
            "path": str(path),
            "state": payload.get("state"),
            "checkpoint": payload.get("checkpoint"),
            "git_commit": payload.get("git_commit"),
            "category_prior_tested": bool(payload.get("category_prior_tested", False)),
        }
    if not statuses:
        raise FileNotFoundError(f"no V9 status artifacts found under {root}")
    if any(row["category_prior_tested"] for row in statuses.values()):
        raise ValueError("V9 status unexpectedly claims that category priors were tested")

    result = {
        "schema": "saga-v10-v9-closeout-v1",
        "passed": True,
        "consumer_git_commit": commit,
        "v9_artifacts_root": str(root),
        "source_statuses": statuses,
        "category_prior_tested": False,
        "withdrawn_conclusion": (
            "V9 did not prove that the learned cross-view identity representation "
            "was insufficient"
        ),
        "retained_conclusion": (
            "V9's specific containment association and core-plus-global-halo "
            "reconstruction failed before category-prior replay"
        ),
        "corrections": {
            "fragment_full_evidence": (
                "V9 final candidates discarded member fragment full_ids and could "
                "collapse a 100-point support to a 3-point core"
            ),
            "pair_score": (
                "V9 containment overlap could score a 3-to-100 fragment pair as 1.0"
            ),
            "edge_diagnostics": (
                "V9 A0/A2 diagnostics contained component proxy pairs and were not "
                "a direct identity-precision measurement"
            ),
        },
        "v10_required_recheck": (
            "real accepted fragment pairs, evidence-preserving full/core funnel, "
            "and view-consensus association"
        ),
    }
    write_json(Path(output_path), result)
    return result


__all__ = ["write_v10_v9_closeout"]
