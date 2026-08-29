from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from category_priors.category_fragment_merge_evaluation import (
    DEV2_SCENE_IDS,
    DEV8_SCENE_IDS,
)
from category_priors.category_fragment_merge_scene_evaluation import (
    evaluate_category_fragment_merge_run,
)


def _manifest(path: Path, scene_ids: tuple[str, ...]) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "scene_runtime_manifest",
                "scenes": [
                    {
                        "scene_id": scene_id,
                        "base_path": str(path.parent / scene_id),
                        "scene_scale_m_per_unit": 1.0,
                    }
                    for scene_id in scene_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _analysis(phase: str) -> dict:
    scene_ids = DEV2_SCENE_IDS if phase == "dev2" else DEV8_SCENE_IDS
    per_scene = [{"scene_id": scene, "candidate_count": 1} for scene in scene_ids]
    return {
        "schema": "saga-category-fragment-merge-evaluation-v1",
        "phase": phase,
        "scene_ids": list(scene_ids),
        "graph_oracle": {"per_scene": per_scene},
        "uniform": {"per_scene": per_scene},
        "class": {"per_scene": per_scene},
        "mechanical_effect": {"per_scene": per_scene},
        "passed": True,
        "conclusion": (
            "dev2-passed-proceed-to-dev8"
            if phase == "dev2"
            else "dev8-passed-category-prior-helps-fragment-assembly"
        ),
    }


def _patch_dependencies(monkeypatch, module, scene_ids: tuple[str, ...]) -> dict:
    dev2_counts = {
        DEV2_SCENE_IDS[0]: 2500,
        DEV2_SCENE_IDS[1]: 2533,
    }
    calls = {
        "loaded": [],
        "written_rows": None,
        "written_json": None,
        "raw_counts": {
            scene_id: dev2_counts.get(scene_id, 1) for scene_id in scene_ids
        },
    }

    def load_scene(path):
        calls["loaded"].append(Path(path).name)
        return SimpleNamespace(
            raw_bank=f"raw-{Path(path).name}",
            graph=f"graph-{Path(path).name}",
            uniform=f"uniform-{Path(path).name}",
            class_shrunk=f"class-{Path(path).name}",
        )

    monkeypatch.setattr(module, "load_category_fragment_scene", load_scene)
    monkeypatch.setattr(
        module,
        "_evaluation_scene",
        lambda **kwargs: f"gt-{kwargs['scene_id']}",
    )

    def evaluate_raw(scene, bank):
        scene_id = str(scene).removeprefix("gt-")
        candidate_count = int(calls["raw_counts"][scene_id])
        payload = {
            "scene_id": scene_id,
            "candidate_count": candidate_count,
            "same_class_iou_025_count": 0,
            "same_class_iou_050_count": 0,
            "candidate_rows": [],
            "best_iou_by_gt": [],
        }
        return SimpleNamespace(
            candidate_count=candidate_count,
            same_class_iou_025_count=0,
            same_class_iou_050_count=0,
            as_dict=lambda payload=payload: dict(payload),
        )

    monkeypatch.setattr(module, "evaluate_cluster_scene", evaluate_raw)
    monkeypatch.setattr(
        module,
        "evaluate_category_fragment_merge",
        lambda **kwargs: _analysis(kwargs["phase"]),
    )
    monkeypatch.setattr(
        module,
        "write_rows",
        lambda path, rows: calls.update(written_rows=(Path(path), list(rows))),
    )
    monkeypatch.setattr(
        module,
        "write_json",
        lambda path, value: calls.update(written_json=(Path(path), value)),
    )
    return calls


def test_dev2_adapter_loads_gt_only_after_complete_runtime_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    import category_priors.category_fragment_merge_scene_evaluation as module

    manifest = _manifest(tmp_path / "runtime.json", DEV2_SCENE_IDS)
    calls = _patch_dependencies(monkeypatch, module, DEV2_SCENE_IDS)

    result = evaluate_category_fragment_merge_run(
        runtime_manifest=manifest,
        gt_dir=tmp_path / "gt",
        run_root=tmp_path / "runs",
        scene_ids=tuple(reversed(DEV2_SCENE_IDS)),
        taxonomy=SimpleNamespace(),
        phase="dev2",
        metrics_output=tmp_path / "metrics.parquet",
        analysis_output=tmp_path / "category_fragment_merge_dev2_analysis.json",
    )

    assert calls["loaded"] == list(DEV2_SCENE_IDS)
    assert result["evaluation_io"]["gt_loaded_only_in_scene_evaluation_adapter"] is True
    assert result["raw_fragment_identity"]["observed"] == {
        "candidate_count": 5033,
        "same_class_iou_025_count": 0,
        "same_class_iou_050_count": 0,
    }
    assert result["raw_fragment_identity"]["passed"] is True
    assert len(calls["written_rows"][1]) == 5 * len(DEV2_SCENE_IDS)
    assert calls["written_json"][1] is result


def test_dev2_raw_fragment_identity_mismatch_blocks_authorization(
    tmp_path: Path, monkeypatch
) -> None:
    import category_priors.category_fragment_merge_scene_evaluation as module

    manifest = _manifest(tmp_path / "runtime.json", DEV2_SCENE_IDS)
    calls = _patch_dependencies(monkeypatch, module, DEV2_SCENE_IDS)
    calls["raw_counts"][DEV2_SCENE_IDS[0]] -= 1

    result = evaluate_category_fragment_merge_run(
        runtime_manifest=manifest,
        gt_dir=tmp_path / "gt",
        run_root=tmp_path / "runs",
        scene_ids=DEV2_SCENE_IDS,
        taxonomy=SimpleNamespace(),
        phase="dev2",
        metrics_output=tmp_path / "metrics.parquet",
        analysis_output=tmp_path / "category_fragment_merge_dev2_analysis.json",
    )

    assert result["raw_fragment_identity"]["passed"] is False
    assert result["passed"] is False
    assert result["conclusion"] == "raw-fragment-identity-mismatch-fix-wiring"


def test_adapter_rejects_nonregistered_scene_set_before_loading(
    tmp_path: Path, monkeypatch
) -> None:
    import category_priors.category_fragment_merge_scene_evaluation as module

    manifest = _manifest(tmp_path / "runtime.json", DEV2_SCENE_IDS)
    calls = _patch_dependencies(monkeypatch, module, DEV2_SCENE_IDS)

    with pytest.raises(ValueError, match="exact registered scene set"):
        evaluate_category_fragment_merge_run(
            runtime_manifest=manifest,
            gt_dir=tmp_path / "gt",
            run_root=tmp_path / "runs",
            scene_ids=(DEV2_SCENE_IDS[0],),
            taxonomy=SimpleNamespace(),
            phase="dev2",
            metrics_output=tmp_path / "metrics.parquet",
            analysis_output=tmp_path / "analysis.json",
        )
    assert calls["loaded"] == []


def test_dev8_requires_passed_sibling_dev2_authorization(
    tmp_path: Path, monkeypatch
) -> None:
    import category_priors.category_fragment_merge_scene_evaluation as module

    manifest = _manifest(tmp_path / "runtime.json", DEV8_SCENE_IDS)
    calls = _patch_dependencies(monkeypatch, module, DEV8_SCENE_IDS)
    output = tmp_path / "category_fragment_merge_dev8_analysis.json"

    with pytest.raises(FileNotFoundError, match="DEV8 requires"):
        evaluate_category_fragment_merge_run(
            runtime_manifest=manifest,
            gt_dir=tmp_path / "gt",
            run_root=tmp_path / "runs",
            scene_ids=DEV8_SCENE_IDS,
            taxonomy=SimpleNamespace(),
            phase="dev8",
            metrics_output=tmp_path / "metrics.parquet",
            analysis_output=output,
        )
    assert calls["loaded"] == []

    authorization = _analysis("dev2")
    (tmp_path / "category_fragment_merge_dev2_analysis.json").write_text(
        json.dumps(authorization), encoding="utf-8"
    )
    result = evaluate_category_fragment_merge_run(
        runtime_manifest=manifest,
        gt_dir=tmp_path / "gt",
        run_root=tmp_path / "runs",
        scene_ids=DEV8_SCENE_IDS,
        taxonomy=SimpleNamespace(),
        phase="dev8",
        metrics_output=tmp_path / "metrics.parquet",
        analysis_output=output,
    )
    assert result["evaluation_io"]["dev2_authorization"]["passed"] is True
    assert result["raw_fragment_identity"]["gate_applies"] is False
    assert result["raw_fragment_identity"]["expected"] is None
    assert result["raw_fragment_identity"]["passed"] is True
    assert calls["loaded"] == list(DEV8_SCENE_IDS)


def test_dev8_rejects_failed_or_mismatched_authorization(
    tmp_path: Path, monkeypatch
) -> None:
    import category_priors.category_fragment_merge_scene_evaluation as module

    manifest = _manifest(tmp_path / "runtime.json", DEV8_SCENE_IDS)
    calls = _patch_dependencies(monkeypatch, module, DEV8_SCENE_IDS)
    authorization = _analysis("dev2")
    authorization["passed"] = False
    auth_path = tmp_path / "category_fragment_merge_dev2_analysis.json"
    auth_path.write_text(json.dumps(authorization), encoding="utf-8")

    with pytest.raises(ValueError, match="does not authorize"):
        evaluate_category_fragment_merge_run(
            runtime_manifest=manifest,
            gt_dir=tmp_path / "gt",
            run_root=tmp_path / "runs",
            scene_ids=DEV8_SCENE_IDS,
            taxonomy=SimpleNamespace(),
            phase="dev8",
            metrics_output=tmp_path / "metrics.parquet",
            analysis_output=tmp_path / "category_fragment_merge_dev8_analysis.json",
        )
    assert calls["loaded"] == []
