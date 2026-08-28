from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from category_priors.final_noise_experiment import (
    candidate_rows,
    load_scene_trace,
    run_final_noise_audit,
)


def _node(area: float) -> dict:
    return {
        "shrunk": {
            "geometry": {
                "log_surface_area_m2": {"q50": float(np.log(area))},
            }
        }
    }


def _write_scene(root: Path, scene: str, counts: dict[int, int]) -> None:
    target = root / scene / "seed-42"
    target.mkdir(parents=True)
    labels = np.concatenate(
        [np.full(count, instance_id, dtype=np.int32) for instance_id, count in counts.items()]
        + [np.asarray([-1], dtype=np.int32)]
    )
    filtered = labels.copy()
    for instance_id, count in counts.items():
        if count < 10:
            filtered[labels == instance_id] = -1
    np.savez_compressed(
        target / "stage_trace.npz",
        post_global_knn=labels,
        post_filter=filtered,
        branch_class_before_merge=labels,
    )
    (target / "stage_trace.json").write_text(
        json.dumps(
            {
                "branch_instance_classes": {
                    str(instance_id): "cup" for instance_id in counts
                }
            }
        ),
        encoding="utf-8",
    )


def test_candidate_rows_keep_the_partition_frozen(tmp_path: Path) -> None:
    _write_scene(tmp_path, "scene", {4: 3, 9: 12})
    trace = load_scene_trace(tmp_path, "scene")
    priors = {"global": _node(4.0), "categories": {"cup": _node(0.04)}}

    rows = candidate_rows(trace, priors)

    assert [(row["instance_id"], row["post_knn_points"]) for row in rows] == [
        (4, 3),
        (9, 12),
    ]
    assert rows[0]["s3_changes_u10"] is True
    assert rows[0]["class_threshold"] == 3
    assert rows[1]["u10_retained"] is True
    assert np.array_equal(trace.post_global_knn, np.array([4, 4, 4] + [9] * 12 + [-1]))


def test_dev_gate_stops_before_confirmation_when_one_scene_has_no_intervention(
    tmp_path: Path,
) -> None:
    traces = tmp_path / "traces"
    _write_scene(traces, "dev_a", {0: 3, 1: 4, 2: 12})
    _write_scene(traces, "dev_b", {0: 12})
    _write_scene(traces, "confirm", {0: 3})
    priors = tmp_path / "priors.json"
    priors.write_text(
        json.dumps({"global": _node(4.0), "categories": {"cup": _node(0.04)}}),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = run_final_noise_audit(
        trace_root=traces,
        prior_json=priors,
        output_root=output,
        scenes=("dev_a", "dev_b", "confirm"),
        dev_scenes=("dev_a", "dev_b"),
    )

    assert result["decision"] == "stop-no-recoverable-final-filter-intervention"
    assert result["dev"]["stopped_at_gate"] == 2
    assert result["dev"]["s3_changed_branch_candidates_by_dev_scene"] == {
        "dev_a": 2,
        "dev_b": 0,
    }
    assert result["confirmation"]["status"] == "not_run"
    assert (output / "noise_threshold_candidates.parquet").is_file()
    assert (output / "noise_threshold_confirm6.parquet").is_file()
