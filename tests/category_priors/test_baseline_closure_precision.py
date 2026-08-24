from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import category_priors.baseline_closure_precision as precision
from category_priors.evaluator import GroundTruthScene


def test_precision_parser_accepts_closeout_paths() -> None:
    args = precision.build_parser().parse_args(
        [
            "--closure-root",
            "closure",
            "--gt-dir",
            "gt",
            "--runtime-manifest",
            "runtime.json",
            "--output-dir",
            "artifacts",
        ]
    )
    assert args.closure_root == Path("closure")
    assert precision.AUDIT_RADII_M == (0.02, 0.05, 0.10)


def test_precision_refuses_to_invent_missing_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(precision, "_output_runs", lambda _root: iter(()))
    with pytest.raises(FileNotFoundError, match="No baseline output"):
        precision.evaluate_teacher_handoff_precision(
            closure_root=tmp_path / "closure",
            gt_dir=tmp_path / "gt",
            runtime_manifest=tmp_path / "runtime.json",
            output_dir=tmp_path / "artifacts",
        )


def test_precision_projects_orphans_and_reports_full_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_json = tmp_path / "output.json"
    output_json.write_text(
        '{"point_labels":[-1,4,9],"instances":{"-1":{"class":"cabinet"},"4":{"class":"chair"}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        precision,
        "_output_runs",
        lambda _root: iter(
            [("full950", "adaptive", "B1-original", "scene", output_json)]
        ),
    )
    monkeypatch.setattr(precision, "_runtime_rows", lambda _path: {"scene": {}})
    monkeypatch.setattr(precision, "_gaussian_ply", lambda _row: tmp_path / "x.ply")
    monkeypatch.setattr(precision, "_transform", lambda _row: np.eye(4))
    xyz = np.asarray(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    monkeypatch.setattr(precision, "load_ply_xyz", lambda _path: xyz)
    monkeypatch.setattr(
        precision,
        "load_ground_truth_npz",
        lambda _path, scene_id: (
            xyz,
            GroundTruthScene(
                scene_id,
                semantic=np.asarray([0, 0, 0]),
                instance=np.asarray([1, 1, 1]),
            ),
        ),
    )
    observed_labels: list[list[int]] = []
    observed_instance_ids: list[list[str]] = []

    def fake_audit(_xyz, labels, instances, *_args, **_kwargs):
        observed_labels.append(labels.tolist())
        observed_instance_ids.append(sorted(instances))
        return {"instances": []}

    monkeypatch.setattr(precision, "evaluate_gaussian_object_precision", fake_audit)
    monkeypatch.setattr(precision, "_select_viewer_cases", lambda _rows: [])
    monkeypatch.setattr(precision, "write_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(precision, "write_json", lambda *_args, **_kwargs: None)

    result = precision.evaluate_teacher_handoff_precision(
        closure_root=tmp_path / "closure",
        gt_dir=tmp_path / "gt",
        runtime_manifest=tmp_path / "runtime.json",
        output_dir=tmp_path / "artifacts",
    )

    assert observed_labels == [[-1, 4, -1], [-1, 4, -1], [-1, 4, -1]]
    assert observed_instance_ids == [["4"], ["4"], ["4"]]
    projection = result["declared_instance_projection"]["runs"][0]
    assert projection["orphan_instance_ids"] == [9]
    assert projection["orphan_counts"] == {"9": 1}
    assert projection["orphan_gaussian_fraction"] == 1 / 3
    assert projection["ignored_negative_metadata_ids"] == [-1]
    assert projection["ignored_negative_metadata_count"] == 1
