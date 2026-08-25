from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import category_priors.v10_runner as v10_runner
from category_priors.v10_runner import (
    V10_STRUCTURE_CONDITIONS,
    load_v10_candidate_bank,
    run_v10_banks,
    v10_object_bank_is_complete,
)
from category_priors.v9_lifting import V9_LIFTING_SCHEMA


def _lifting_root(tmp_path: Path, scene_ids: tuple[str, ...]) -> Path:
    root = tmp_path / "lifting"
    for scene_id in scene_ids:
        target = root / scene_id
        target.mkdir(parents=True)
        (target / "lifting_bank.json").write_text(
            json.dumps(
                {
                    "schema": V9_LIFTING_SCHEMA,
                    "scene_id": scene_id,
                    "point_count": 13,
                    "frame_count": 2,
                    "identity": {
                        "schema": "test-lifting-identity",
                        "scene_id": scene_id,
                        "git_commit": "producer",
                    },
                }
            ),
            encoding="utf-8",
        )
    return root


def _candidate(condition: str) -> dict[str, Any]:
    row = {
        "candidate_id": 0,
        "track_id": 7,
        "branch_class": "book",
        "classification_eligible": True,
        "full_point_count": 12,
        "core_point_count": 10,
        "base_score": 0.9,
        "metric_extents_m": [0.1, 0.2, 0.3],
        "local_surface_density": 10.0,
        "boundary_ratio_5cm": 0.2,
        "structure_condition": condition,
    }
    row["classifiers"] = {
        classifier: {
            "branch_class": "book",
            "class_id": 0,
            "semantic_ratio": 0.9,
            "classification_eligible": True,
        }
        for classifier in ("mv-label", "codebook")
    }
    return row


def _payload(condition: str) -> dict[str, Any]:
    stages = {
        stage: [
            {
                "candidate_id": 0,
                "class_name": "book" if stage == "final_candidate" else None,
                "gaussian_ids": np.arange(10, dtype=np.int32),
            }
        ]
        for stage in v10_runner.V10_FUNNEL_STAGES
    }
    return {
        "point_count": 13,
        "fragments": [{"fragment_id": 2, "frame_id": 0}],
        "tracks": [{"track_id": 7, "fragment_ids": [2]}],
        "candidates": [_candidate(condition)],
        "full_ids": [np.arange(12, dtype=np.int32)],
        "core_ids": [np.arange(10, dtype=np.int32)],
        "accepted_edges": [],
        "stage_supports": stages,
        "diagnostics": {"condition": condition},
    }


def _payload_with_valid_edge(condition: str = "VC1") -> dict[str, Any]:
    payload = _payload(condition)
    payload["fragments"] = [
        {"fragment_id": 2, "frame_id": 0},
        {"fragment_id": 3, "frame_id": 1},
    ]
    payload["tracks"] = [
        {"track_id": 7, "fragment_ids": [2, 3], "frame_ids": [0, 1]}
    ]
    payload["accepted_edges"] = [
        {
            "left_fragment_id": 2,
            "right_fragment_id": 3,
            "left_frame_id": 0,
            "right_frame_id": 1,
            "kind": "strong",
            "score": 0.9,
            "shared": 3,
            "strong": True,
            "cycle_supported": False,
            "frame_weighted_jaccard": 0.7,
            "p0_overlap": 0.4,
            "left_coverage": 0.9,
            "right_coverage": 0.8,
            "row_margin": 0.2,
            "column_margin": 0.2,
            "component_support_ratio": 0.9,
        }
    ]
    return payload


def test_two_scene_five_arm_runner_is_read_only_compact_and_resumable(
    tmp_path: Path, monkeypatch,
) -> None:
    scenes = ("scene0000_00", "scene0001_00")
    lifting_root = _lifting_root(tmp_path, scenes)
    original_headers = {
        scene: (lifting_root / scene / "lifting_bank.json").read_bytes()
        for scene in scenes
    }
    calls: list[tuple[str, str]] = []

    def fake_load(source: Path):
        metadata = json.loads((source / "lifting_bank.json").read_text("utf-8"))
        return metadata, {"xyz_m": np.zeros((13, 3), dtype=np.float32)}

    def fake_builder(metadata, arrays, *, condition: str):
        scene_id = str(metadata["scene_id"])
        calls.append((scene_id, condition))
        assert not arrays["xyz_m"].flags.writeable
        try:
            arrays["xyz_m"][0, 0] = 1
        except ValueError:
            pass
        else:  # pragma: no cover - proves the read-only adapter is active
            raise AssertionError("V9 lifting arrays must be read-only")
        return _payload(condition)

    monkeypatch.setattr(v10_runner, "load_lifting_bank", fake_load)
    output_root = tmp_path / "banks"
    summary = run_v10_banks(
        lifting_root=lifting_root,
        output_root=output_root,
        scene_ids=scenes,
        git_commit="consumer",
        builder=fake_builder,
    )
    assert len(summary["runs"]) == 10
    assert set(calls) == {
        (scene, condition) for scene in scenes for condition in V10_STRUCTURE_CONDITIONS
    }
    for scene in scenes:
        assert (lifting_root / scene / "lifting_bank.json").read_bytes() == original_headers[scene]
        for condition in V10_STRUCTURE_CONDITIONS:
            target = output_root / condition / scene
            assert v10_object_bank_is_complete(
                target,
                expected_scene_id=scene,
                expected_condition=condition,
                expected_source_lifting=lifting_root / scene,
                expected_git_commit="consumer",
            )
            metadata, bank = load_v10_candidate_bank(target)
            assert metadata["candidate_count"] == 1
            assert bank.full_ids[0].tolist() == list(range(12))
            assert bank.core_ids[0].tolist() == list(range(10))
            assert not bank.full_ids[0].flags.writeable

    def forbidden(*_args, **_kwargs):
        raise AssertionError("complete V10 banks must skip lifting and builder")

    monkeypatch.setattr(v10_runner, "load_lifting_bank", forbidden)
    resumed = run_v10_banks(
        lifting_root=lifting_root,
        output_root=output_root,
        scene_ids=scenes,
        git_commit="consumer",
        builder=forbidden,
    )
    assert {row["status"] for row in resumed["runs"]} == {"reused"}

    damaged = output_root / "VC1" / scenes[0] / "object_bank.npz"
    damaged.write_bytes(b"damaged")
    repaired: list[str] = []

    def one_load(source: Path):
        metadata = json.loads((source / "lifting_bank.json").read_text("utf-8"))
        return metadata, {"xyz_m": np.zeros((13, 3), dtype=np.float32)}

    def one_builder(_metadata, _arrays, *, condition: str):
        repaired.append(condition)
        return _payload(condition)

    monkeypatch.setattr(v10_runner, "load_lifting_bank", one_load)
    run_v10_banks(
        lifting_root=lifting_root,
        output_root=output_root,
        scene_ids=scenes,
        git_commit="consumer",
        builder=one_builder,
    )
    assert repaired == ["VC1"]
    assert v10_object_bank_is_complete(damaged.parent)


def test_runner_rejects_builder_core_outside_full(tmp_path: Path, monkeypatch) -> None:
    lifting_root = _lifting_root(tmp_path, ("scene0000_00",))

    def fake_load(source: Path):
        metadata = json.loads((source / "lifting_bank.json").read_text("utf-8"))
        return metadata, {"xyz_m": np.zeros((13, 3), dtype=np.float32)}

    def invalid_builder(_metadata, _arrays, *, condition: str):
        payload = _payload(condition)
        payload["core_ids"] = [np.asarray([12], dtype=np.int32)]
        payload["candidates"][0]["core_point_count"] = 1
        return payload

    monkeypatch.setattr(v10_runner, "load_lifting_bank", fake_load)
    try:
        run_v10_banks(
            lifting_root=lifting_root,
            output_root=tmp_path / "banks",
            scene_ids=["scene0000_00"],
            conditions=["VC1"],
            git_commit="consumer",
            builder=invalid_builder,
        )
    except ValueError as exc:
        assert "subset" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid candidate geometry must be rejected")


def test_runner_accepts_only_real_thresholded_edge_evidence() -> None:
    payload = _payload_with_valid_edge()
    normalized, _ = v10_runner._normalise_builder_payload(
        payload, point_count=13, condition="VC1"
    )
    assert normalized["accepted_edges"] == payload["accepted_edges"]

    invalid_payloads: list[dict[str, Any]] = []

    unknown = copy.deepcopy(payload)
    unknown["accepted_edges"][0]["right_fragment_id"] = 99
    invalid_payloads.append(unknown)

    wrong_frame = copy.deepcopy(payload)
    wrong_frame["accepted_edges"][0]["right_frame_id"] = 0
    invalid_payloads.append(wrong_frame)

    duplicate = copy.deepcopy(payload)
    duplicate["accepted_edges"].append(copy.deepcopy(duplicate["accepted_edges"][0]))
    invalid_payloads.append(duplicate)

    weak_noncycle = copy.deepcopy(payload)
    weak_noncycle["accepted_edges"][0].update(
        {"strong": False, "cycle_supported": False, "kind": "pair"}
    )
    invalid_payloads.append(weak_noncycle)

    subthreshold_strong = copy.deepcopy(payload)
    subthreshold_strong["accepted_edges"][0]["left_coverage"] = 0.79
    invalid_payloads.append(subthreshold_strong)

    proxy_edge = copy.deepcopy(payload)
    proxy_edge["tracks"] = [
        {"track_id": 7, "fragment_ids": [2], "frame_ids": [0]},
        {"track_id": 8, "fragment_ids": [3], "frame_ids": [1]},
    ]
    invalid_payloads.append(proxy_edge)

    cyclic = copy.deepcopy(payload)
    cyclic["fragments"].append({"fragment_id": 4, "frame_id": 2})
    cyclic["tracks"] = [
        {"track_id": 7, "fragment_ids": [2, 3, 4], "frame_ids": [0, 1, 2]}
    ]
    for left, right, left_frame, right_frame in ((3, 4, 1, 2), (4, 2, 2, 0)):
        edge = copy.deepcopy(payload["accepted_edges"][0])
        edge.update(
            {
                "left_fragment_id": left,
                "right_fragment_id": right,
                "left_frame_id": left_frame,
                "right_frame_id": right_frame,
            }
        )
        cyclic["accepted_edges"].append(edge)
    invalid_payloads.append(cyclic)

    for invalid in invalid_payloads:
        with pytest.raises(ValueError):
            v10_runner._normalise_builder_payload(
                invalid, point_count=13, condition="VC1"
            )
