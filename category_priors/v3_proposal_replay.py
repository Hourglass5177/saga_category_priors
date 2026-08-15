from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import load_json, write_json


def candidate_acceptance(
    candidate: Mapping[str, Any],
    *,
    background_points: int,
    minimum_points: int = 100,
) -> tuple[bool, str]:
    """Apply one fixed, conservative proposal-level acceptance rule."""
    total = int(candidate.get("active_branch_points", 0))
    vote = candidate.get("vote", {})
    if total < minimum_points or background_points < minimum_points:
        return False, "too_small"
    if not bool(vote.get("winner_matches_branch", False)):
        return False, "vote_class_mismatch"
    vote_ratio = float(vote.get("branch_class_ratio", 0.0))
    background_ratio = float(vote.get("background_ratio", 1.0))
    if vote_ratio < 0.60 or vote_ratio <= background_ratio:
        return False, "weak_vote"
    if float(candidate.get("assignment_confidence_mean", 0.0)) < 0.80:
        return False, "weak_assignment"
    membership = candidate.get("hdbscan_membership_mean")
    if membership is None or float(membership) < 0.70:
        return False, "weak_membership"
    persistence = candidate.get("hdbscan_persistence")
    if persistence is None or float(persistence) < 0.03:
        return False, "weak_persistence"
    overlap = candidate.get("global_final_overlap", {})
    if float(overlap.get("fraction", 1.0)) > 0.50:
        return False, "overlaps_b1"
    if background_points / max(total, 1) < 0.50:
        return False, "insufficient_background"
    return True, "accepted"


def proposal_score(candidate: Mapping[str, Any]) -> float:
    vote = float(candidate["vote"]["branch_class_ratio"])
    assignment = float(candidate["assignment_confidence_mean"])
    membership = float(candidate["hdbscan_membership_mean"])
    persistence = float(candidate["hdbscan_persistence"])
    score = vote * assignment * membership * (0.5 + 0.5 * persistence)
    return float(np.clip(score, 0.0, 1.0))


def replay_scene(
    *,
    scene_id: str,
    seed: int,
    base_run_dir: Path,
    shadow_run_dir: Path,
    output_run_dir: Path,
    condition: str,
    mode: str = "exclusive",
) -> dict[str, Any]:
    output = load_json(base_run_dir / "output.json")
    diagnostics = load_json(base_run_dir / "diagnostics.json")
    shadow = load_json(shadow_run_dir / f"shadow-{mode}.json")
    arrays = np.load(shadow_run_dir / f"branch-labels-{mode}.npz")
    branch_labels = np.asarray(arrays["branch_labels"], dtype=np.int64)
    labels = np.asarray(output["point_labels"], dtype=np.int64)
    if labels.shape != branch_labels.shape:
        raise ValueError(f"{scene_id}: B1 and shadow labels have different lengths")

    instances = {str(key): dict(value) for key, value in output["instances"].items()}
    metadata_instances = {
        str(key): dict(value)
        for key, value in diagnostics.get("instances", {}).items()
    }
    next_instance = max((int(key) for key in instances), default=-1) + 1
    reasons: dict[str, int] = {}
    accepted: list[dict[str, Any]] = []

    for candidate in sorted(
        shadow.get("candidates", []),
        key=lambda item: (-proposal_score(item), int(item["candidate_id"])),
    ):
        candidate_id = int(candidate["candidate_id"])
        candidate_mask = branch_labels == candidate_id
        background_mask = candidate_mask & (labels < 0)
        background_points = int(background_mask.sum())
        keep, reason = candidate_acceptance(
            candidate, background_points=background_points
        )
        reasons[reason] = reasons.get(reason, 0) + 1
        if not keep:
            continue
        class_name = str(candidate["branch_class"])
        labels[background_mask] = next_instance
        score = proposal_score(candidate)
        instances[str(next_instance)] = {"class": class_name}
        metadata_instances[str(next_instance)] = {
            "class": class_name,
            "score": score,
            "source": "v3_proposal_replay",
            "candidate_id": candidate_id,
        }
        accepted.append(
            {
                "candidate_id": candidate_id,
                "instance_id": next_instance,
                "class": class_name,
                "points": background_points,
                "score": score,
            }
        )
        next_instance += 1

    output["point_labels"] = labels.tolist()
    output["instances"] = instances
    diagnostics["instances"] = metadata_instances
    diagnostics["status"] = "complete"
    diagnostics["kind"] = "teacher_prior_diagnostics"
    diagnostics["run"] = {
        **dict(diagnostics.get("run", {})),
        "condition": condition,
        "scene_id": scene_id,
        "seed": int(seed),
    }
    diagnostics["runner"] = {
        **dict(diagnostics.get("runner", {})),
        "point_count": int(len(labels)),
        "instance_count": int(len(instances)),
    }
    diagnostics["v3_proposal_replay"] = {
        "mode": mode,
        "accepted_count": len(accepted),
        "accepted_points": int(sum(item["points"] for item in accepted)),
        "rejection_reasons": reasons,
        "accepted": accepted,
    }
    output_run_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_run_dir / "output.json", output)
    write_json(output_run_dir / "diagnostics.json", diagnostics)
    return diagnostics["v3_proposal_replay"]


def run_replay(
    *,
    base_root: str | Path,
    shadow_root: str | Path,
    output_root: str | Path,
    scenes: Sequence[str],
    seed: int,
    condition: str,
    mode: str,
) -> dict[str, Any]:
    base_root = Path(base_root)
    shadow_root = Path(shadow_root)
    output_root = Path(output_root)
    per_scene = {}
    for scene_id in scenes:
        per_scene[scene_id] = replay_scene(
            scene_id=scene_id,
            seed=seed,
            base_run_dir=base_root / "original" / scene_id / f"seed-{seed}",
            shadow_run_dir=shadow_root / scene_id / f"seed-{seed}",
            output_run_dir=output_root / condition / scene_id / f"seed-{seed}",
            condition=condition,
            mode=mode,
        )
    return {"condition": condition, "seed": seed, "scenes": per_scene}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--shadow-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition", default="B1-proposal-protected")
    parser.add_argument("--mode", choices=("exact", "exclusive"), default="exclusive")
    parser.add_argument("--summary-output")
    args = parser.parse_args()
    result = run_replay(
        base_root=args.base_root,
        shadow_root=args.shadow_root,
        output_root=args.output_root,
        scenes=args.scene,
        seed=args.seed,
        condition=args.condition,
        mode=args.mode,
    )
    if args.summary_output:
        write_json(args.summary_output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
