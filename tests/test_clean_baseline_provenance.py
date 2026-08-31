from __future__ import annotations

from category_priors.clean_baseline.provenance import (
    audit_teacher_mask_roles,
    build_clean_baseline_provenance,
)


def test_teacher_mask_role_audit_distinguishes_supervision_from_identity() -> None:
    roles = audit_teacher_mask_roles(
        training_source="original_masks sam_masks contrastive_loss",
        postprocess_source=(
            "HDBSCAN cluster_centers confidence.max NearestNeighbors knn "
            "filter_num then load masks_path for vote"
        ),
    )
    assert roles == {
        "masks_supervise_features": True,
        "automatic_instances_use_hdbscan": True,
        "automatic_instances_use_center_assignment": True,
        "automatic_instances_use_global_smoothing": True,
        "masks_vote_after_instance_formation": True,
        "persistent_frame_mask_identity_in_instance_partition": False,
    }


def test_provenance_records_why_legacy_sam_alpha_bank_is_not_reused(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "category_priors.clean_baseline.provenance._git_text",
        lambda _root, _revision, path: (
            "original_masks sam_masks"
            if path == "train_contrastive_feature.py"
            else (
                "HDBSCAN cluster_centers confidence.max NearestNeighbors knn "
                "filter_num then load masks_path for vote"
            )
        ),
    )
    monkeypatch.setattr(
        "category_priors.clean_baseline.provenance._git_head",
        lambda _root: "registered-commit",
    )

    result = build_clean_baseline_provenance(
        repo_root=tmp_path,
        tracked_dirty_reader=lambda _root: (),
    )

    compatibility = result["legacy_sam_alpha_bank_compatibility"]
    assert compatibility["compatible"] is False
    assert "complete SAM observation set" in compatibility["reason"]
    assert compatibility["legacy_artifacts_mutated"] is False
    assert (
        result["independent_implementation"]["reference_repository_head_on_2026_08_31"]
        == "eb2d41c2267cde21966f4340c37b8cf94b05c23c"
    )
    assert result["tracked_worktree_clean"] is True


def test_provenance_rejects_tracked_dirty_deployment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "category_priors.clean_baseline.provenance._git_text",
        lambda *_args: "unused",
    )
    monkeypatch.setattr(
        "category_priors.clean_baseline.provenance._git_head",
        lambda _root: "a" * 40,
    )
    try:
        build_clean_baseline_provenance(
            repo_root=tmp_path,
            tracked_dirty_reader=lambda _root: (" M tracked.py",),
        )
    except RuntimeError as exc:
        assert "clean tracked worktree" in str(exc)
    else:  # pragma: no cover - makes the expected deployment refusal explicit
        raise AssertionError("tracked dirty worktree was accepted")
