from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import category_priors.cli as cli


def _commands() -> set[str]:
    parser = cli.build_parser()
    return set(parser._subparsers._group_actions[0].choices)


def test_public_cli_exposes_only_reusable_commands() -> None:
    assert _commands() == {
        "fit",
        "evaluate",
        "prepare-gt",
        "audit-saga-alignment",
        "audit-gaussian-objects",
    }


def test_fit_and_evaluate_dispatch(monkeypatch, tmp_path: Path) -> None:
    taxonomy = object()
    fitted = {"kind": "priors"}
    calls: dict[str, object] = {}
    monkeypatch.setattr(cli, "read_rows", lambda path: [{"source": path}])
    monkeypatch.setattr(cli, "load_taxonomy", lambda path: taxonomy)

    def fake_fit(rows, actual_taxonomy, source, **kwargs):
        calls["fit"] = (rows, actual_taxonomy, source, kwargs)
        return fitted

    monkeypatch.setattr(cli, "fit_priors", fake_fit)
    monkeypatch.setattr(
        cli,
        "write_priors",
        lambda path, payload: calls.__setitem__("write", (path, payload)),
    )
    cli.main(
        [
            "fit",
            "--stats",
            "stats.parquet",
            "--output",
            "priors.json",
            "--bootstrap-samples",
            "12",
        ]
    )
    assert calls["fit"] == (
        [{"source": "stats.parquet"}],
        taxonomy,
        "stats.parquet",
        {
            "seed": 20260804,
            "bootstrap_samples": 12,
            "min_physical_scenes": 5,
            "shrink_tau": 20.0,
        },
    )
    assert calls["write"] == ("priors.json", fitted)

    def fake_evaluate(*args):
        calls["evaluate"] = args

    monkeypatch.setattr(cli, "evaluate_manifest", fake_evaluate)
    cli.main(
        [
            "evaluate",
            "--manifest",
            "evaluation.json",
            "--output",
            str(tmp_path / "metrics.json"),
        ]
    )
    assert calls["evaluate"] == (
        "evaluation.json",
        taxonomy,
        str(tmp_path / "metrics.json"),
        0.05,
        100,
    )


def test_prepare_gt_writes_canonical_arrays_and_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("scene0001_00\nscene0002_00\n", encoding="utf-8")
    output = tmp_path / "gt"
    taxonomy = SimpleNamespace(content_hash="taxonomy-hash")
    monkeypatch.setattr(cli, "load_taxonomy", lambda _path: taxonomy)
    monkeypatch.setattr(
        cli,
        "discover_scene_files",
        lambda root, scene: SimpleNamespace(root=root, scene_id=scene),
    )

    def fake_prepare(files, actual_taxonomy, dataset):
        assert actual_taxonomy is taxonomy
        assert dataset == "scannet200"
        value = 1 if files.scene_id == "scene0001_00" else 2
        return (
            np.asarray([[value, 0.0, 0.0], [value, 1.0, 0.0]]),
            np.asarray([0, -1]),
            np.asarray([value, -1]),
        )

    monkeypatch.setattr(cli, "prepare_scene_ground_truth", fake_prepare)
    cli.main(
        [
            "prepare-gt",
            "--dataset-root",
            "dataset",
            "--scene-list",
            str(scene_list),
            "--output-dir",
            str(output),
        ]
    )

    with np.load(output / "scene0001_00.npz") as arrays:
        assert arrays["coords"].shape == (2, 3)
        assert arrays["semantic"].tolist() == [0, -1]
        assert arrays["instance"].tolist() == [1, -1]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "canonical_ground_truth"
    assert manifest["taxonomy_sha256"] == "taxonomy-hash"
    assert [row["scene_id"] for row in manifest["scenes"]] == [
        "scene0001_00",
        "scene0002_00",
    ]
    assert [row["mapped_vertices"] for row in manifest["scenes"]] == [1, 1]
    assert not list(output.glob("*.part"))


def test_alignment_and_object_audits_dispatch(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    def fake_alignment(*args, **kwargs):
        calls["alignment"] = (args, kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(cli, "audit_saga_alignment", fake_alignment)
    cli.main(
        [
            "audit-saga-alignment",
            "--preparation-manifest",
            "prepared.json",
            "--gt-npz",
            "gt.npz",
            "--gaussian-ply",
            "gaussians.ply",
            "--output",
            "alignment.json",
            "--minimal",
        ]
    )
    assert calls["alignment"] == (
        ("prepared.json", "gt.npz", "alignment.json"),
        {
            "gaussian_ply_path": "gaussians.ply",
            "radius_m": 0.05,
            "minimum_mapped_fraction": 0.90,
            "camera_padding_m": 2.0,
            "minimal": True,
        },
    )

    taxonomy = object()
    monkeypatch.setattr(cli, "load_taxonomy", lambda _path: taxonomy)

    def fake_object_audit(**kwargs):
        calls["objects"] = kwargs
        return {"status": "complete"}

    monkeypatch.setattr(cli, "audit_gaussian_object_runs", fake_object_audit)
    cli.main(
        [
            "audit-gaussian-objects",
            "--scene-manifest",
            "scenes.json",
            "--gt-dir",
            "gt",
            "--runs-root",
            "runs",
            "--scene",
            "scene0001_00",
            "--condition",
            "B0",
            "--table-output",
            "rows.parquet",
            "--audit-output",
            "audit.json",
            "--comparison-output",
            "compare.json",
            "--viewer-output",
            "viewer",
        ]
    )
    assert calls["objects"] == {
        "scene_manifest": "scenes.json",
        "gt_dir": "gt",
        "runs_root": "runs",
        "taxonomy": taxonomy,
        "scene_ids": ["scene0001_00"],
        "conditions": ["B0"],
        "seed": 42,
        "table_output": "rows.parquet",
        "audit_output": "audit.json",
        "comparison_output": "compare.json",
        "viewer_output": "viewer",
        "radius_m": 0.05,
        "min_region_size": 100,
    }
    stdout = capsys.readouterr().out
    assert '"status": "ok"' in stdout
    assert '"status": "complete"' in stdout
