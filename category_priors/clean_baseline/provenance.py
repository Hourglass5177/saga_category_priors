from __future__ import annotations

"""Machine-readable provenance for the clean Gaussian--mask baseline."""

import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..io import write_json


TEACHER_HANDOFF_CANDIDATE = "8c5e167"


def audit_teacher_mask_roles(
    *,
    training_source: str,
    postprocess_source: str,
) -> dict[str, bool]:
    """Classify the roles of masks in the teacher-handoff candidate.

    This is deliberately a source-level audit rather than a behavioural claim:
    it records that masks supervise learned features and late category voting.
    Their information is therefore indirectly compressed into learned
    similarity features; what is not retained by the automatic instance path
    is the exact per-frame mask membership and an explicit cross-view object
    correspondence graph.  The partition is reconstructed later by clustering
    and smoothing those Gaussian-level features.
    """

    training = str(training_source)
    postprocess = str(postprocess_source)
    return {
        "masks_supervise_features": (
            "original_masks" in training and "sam_masks" in training
        ),
        "automatic_instances_use_hdbscan": "HDBSCAN" in postprocess,
        "automatic_instances_use_center_assignment": (
            "cluster_centers" in postprocess
            and ("confidence.max" in postprocess or "point_labels" in postprocess)
        ),
        "automatic_instances_use_global_smoothing": (
            "filter_num" in postprocess
            and ("NearestNeighbors" in postprocess or "knn" in postprocess.lower())
        ),
        "masks_vote_after_instance_formation": (
            "masks_path" in postprocess
            and "filter_num" in postprocess
            and postprocess.find("filter_num") < postprocess.rfind("masks_path")
        ),
        "persistent_frame_mask_identity_in_instance_partition": False,
    }


def _git_text(repo_root: Path, revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _git_tracked_dirty(repo_root: Path) -> tuple[str, ...]:
    """Return tracked staged/unstaged changes, excluding user untracked files."""

    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return tuple(line for line in result.stdout.splitlines() if line.strip())


def build_clean_baseline_provenance(
    *,
    repo_root: str | Path,
    output_path: str | Path | None = None,
    teacher_revision: str = TEACHER_HANDOFF_CANDIDATE,
    tracked_dirty_reader: Callable[[Path], Sequence[str]] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    reader = _git_tracked_dirty if tracked_dirty_reader is None else tracked_dirty_reader
    tracked_dirty = tuple(map(str, reader(root)))
    if tracked_dirty:
        raise RuntimeError(
            "clean-baseline deployment requires a clean tracked worktree; "
            f"found {len(tracked_dirty)} staged/unstaged tracked path(s)"
        )
    roles = audit_teacher_mask_roles(
        training_source=_git_text(root, teacher_revision, "train_contrastive_feature.py"),
        postprocess_source=_git_text(root, teacher_revision, "postprocess.py"),
    )
    if not all(value for key, value in roles.items() if key != "persistent_frame_mask_identity_in_instance_partition"):
        raise RuntimeError("teacher-handoff mask-role audit did not match the registered architecture")
    payload: dict[str, Any] = {
        "schema": "saga-clean-baseline-provenance-v1",
        "current_commit": _git_head(root),
        "tracked_worktree_clean": True,
        "untracked_files_ignored_by_deployment_gate": True,
        "teacher_handoff_candidate": teacher_revision,
        "teacher_handoff_is_byte_exact_dirty_snapshot": False,
        "teacher_mask_roles": roles,
        "system_name": "SAGA-asset Gaussian-mask consensus baseline",
        "official_saga_scope": "promptable 3D Gaussian segmentation, not an automatic ScanNet instance-AP baseline",
        "independent_implementation": {
            "method_reference": "MaskClustering CVPR 2024 paper",
            "paper_version": "CVPR 2024 open-access proceedings",
            "reference_repository": "https://github.com/PKU-EPIC/MaskClustering",
            "reference_repository_head_on_2026_08_31": (
                "eb2d41c2267cde21966f4340c37b8cf94b05c23c"
            ),
            "source_code_copied": False,
            "reference_repository_license_detected": False,
        },
        "legacy_sam_alpha_bank_compatibility": {
            "compatible": False,
            "reason": (
                "the legacy S-AM bank stores only masks that passed its old "
                "M1-core fragment gate, so the complete SAM observation set "
                "cannot be reconstructed without rendering"
            ),
            "legacy_artifacts_mutated": False,
        },
        "references": {
            "official_saga": "https://github.com/Jumpat/SegAnyGAussians",
            "official_saga_head_on_2026_08_31": (
                "2d4c5d77c857c956d747e4775d3d72c4ec5dfe16"
            ),
            "maskclustering_paper": "https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_MaskClustering_View_Consensus_based_Mask_Graph_Clustering_for_Open-Vocabulary_3D_CVPR_2024_paper.pdf",
        },
    }
    if output_path is not None:
        write_json(output_path, payload)
    return payload


__all__ = [
    "TEACHER_HANDOFF_CANDIDATE",
    "audit_teacher_mask_roles",
    "build_clean_baseline_provenance",
]
