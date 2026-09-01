from __future__ import annotations

import json
from pathlib import Path

import pytest

from category_priors.clean_baseline import materialize_two_step_manifest as module
from category_priors.io import hash_json, sha256_file


SCENES = module.REGISTERED_DEV8_SCENE_IDS
CONDITIONS = ("C0-no-prior", "U-global")


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _signed_identity(scene_id: str, condition: str) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema": "saga-clean-alpha-mask-run-identity-v1",
        "consumer_commit": "a" * 40,
        "scene_id": scene_id,
        "condition": condition,
    }
    identity["content_sha256"] = hash_json(identity)
    return identity


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    relative: bool,
) -> dict[str, Path]:
    legacy_root = tmp_path / "legacy"
    import_root = tmp_path / "imports"
    legacy_root.mkdir()
    import_root.mkdir()
    legacy_scenes = []
    imported: dict[str, dict[str, object]] = {}
    evaluation_inputs: dict[str, dict[str, object]] = {}
    first_paths: dict[str, Path] = {}
    for scene_id in SCENES:
        gt = _write(legacy_root / f"{scene_id}-gt.json", {})
        ply = _write(legacy_root / f"{scene_id}.ply", {})
        bank = import_root / f"{scene_id}-bank"
        bank.mkdir()
        request = _write(
            import_root / f"{scene_id}-request.json",
            {
                "schema": "saga-clean-alpha-mask-evidence-request-v1",
                "producer_commit": "1" * 40,
                "scene": {"scene_id": scene_id},
            },
        )
        outputs: dict[str, str] = {}
        prediction_ids: dict[str, str] = {}
        last_output = request
        for condition in CONDITIONS:
            directory = legacy_root / condition / scene_id
            identity = _signed_identity(scene_id, condition)
            last_output = _write(
                directory / "output.json",
                {
                    "scene_id": scene_id,
                    "condition": condition,
                    "run_identity": identity,
                },
            )
            _write(
                directory / "diagnostics.json",
                {
                    "scene_id": scene_id,
                    "condition": condition,
                    "run_identity": identity,
                },
            )
            outputs[condition] = (
                str(last_output.relative_to(legacy_root))
                if relative
                else str(last_output)
            )
            prediction_ids[condition] = str(identity["content_sha256"])
        transform = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        legacy_scenes.append(
            {
                "scene_id": scene_id,
                "gt_npz": gt.name if relative else str(gt),
                "gaussian_ply": ply.name if relative else str(ply),
                "gaussian_to_gt_transform": transform,
                "outputs": outputs,
            }
        )
        imported[scene_id] = {
            "bank_dir": bank,
            "source_request": request,
            "producer_commit": "1" * 40,
            "files": {
                "evidence.npz": "2" * 64,
                "masks.json": "3" * 64,
                "diagnostics.json": "4" * 64,
            },
            "request": json.loads(request.read_text(encoding="utf-8")),
        }
        evaluation_inputs[scene_id] = {
            "predictions": prediction_ids,
            "gt_sha256": sha256_file(gt),
            "gaussian_sha256": sha256_file(ply),
            "gaussian_to_gt_transform": transform,
        }
        if scene_id == SCENES[0]:
            first_paths = {"gt": gt, "ply": ply, "bank": bank, "output": last_output}

    legacy = _write(
        legacy_root / "manifest.json",
        {"kind": "clean_baseline_evaluation_manifest", "scenes": legacy_scenes},
    )
    import_manifest = _write(
        import_root / "manifest.json",
        {"schema": "saga-clean-evidence-import-manifest-v1", "scenes": {}},
    )
    monkeypatch.setattr(module, "_load_evidence_imports", lambda _: imported)
    evaluation_identity: dict[str, object] = {
        "schema": "saga-clean-alpha-mask-evaluation-identity-v1",
        "manifest": "5" * 64,
        "class_names": list(module.load_taxonomy().canonical_classes),
        "conditions": list(CONDITIONS),
        "radius_m": 0.05,
        "minimum_mapped_fraction": 0.9,
        "min_region_size": 100,
        "inputs": evaluation_inputs,
    }
    evaluation_identity["content_sha256"] = hash_json(evaluation_identity)
    historical = _write(
        tmp_path / "historical.json",
        {
            "schema": "saga-clean-alpha-mask-evaluation-v2",
            "scene_ids": list(SCENES),
            "conditions": list(CONDITIONS),
            "radius_m": 0.05,
            "minimum_mapped_fraction": 0.9,
            "min_region_size": 100,
            "oracle_class_in_formal_metrics": False,
            "evaluation_identity": evaluation_identity,
            "metrics": {
                condition: {
                    "aggregate": {
                        "map_0.25": 0.1,
                        "map_0.50": 0.02,
                        "map_50_95": 0.01,
                    }
                }
                for condition in CONDITIONS
            },
        },
    )
    sizes = _write(
        tmp_path / "sizes.json",
        {
            "boundaries_m": {
                "tiny_max_m": 0.5,
                "small_max_m": 1.0,
                "medium_max_m": 2.0,
            }
        },
    )
    return {
        "legacy": legacy,
        "imports": import_manifest,
        "historical": historical,
        "sizes": sizes,
        **first_paths,
    }


def test_materialize_strict_manifest_from_frozen_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, relative=False)
    output = tmp_path / "manifest.json"
    payload = module.materialize_two_step_manifest(
        legacy_evaluation_manifest=paths["legacy"],
        evidence_import_manifest=paths["imports"],
        historical_evaluation=paths["historical"],
        size_bins=paths["sizes"],
        output_path=output,
        dev2_scene_ids=SCENES[:2],
        dev8_scene_ids=SCENES,
    )
    assert payload["schema"] == "saga-clean-mask-contract-manifest-v1"
    assert payload["dev8_scene_ids"] == list(SCENES)
    assert payload["min_region_size"] == 100
    assert len(payload["scenes"]) == 8
    assert payload["scenes"][0]["conditions"]["C0-no-prior"][
        "diagnostics"
    ].endswith("diagnostics.json")
    assert json.loads(output.read_text()) == payload


def test_materialize_resolves_paths_from_registering_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, relative=True)
    payload = module.materialize_two_step_manifest(
        legacy_evaluation_manifest=paths["legacy"],
        evidence_import_manifest=paths["imports"],
        historical_evaluation=paths["historical"],
        size_bins=paths["sizes"],
        output_path=tmp_path / "out.json",
        dev2_scene_ids=SCENES[:2],
        dev8_scene_ids=SCENES,
    )
    row = payload["scenes"][0]
    assert row["gt_npz"] == str(paths["gt"].resolve())
    assert row["gaussian_ply"] == str(paths["ply"].resolve())
    assert row["bank_dir"] == str(paths["bank"].resolve())


def test_historical_prediction_identity_must_match_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, relative=False)
    payload = json.loads(paths["historical"].read_text(encoding="utf-8"))
    payload["evaluation_identity"]["inputs"][SCENES[0]]["predictions"][
        "C0-no-prior"
    ] = "f" * 64
    unsigned = dict(payload["evaluation_identity"])
    unsigned.pop("content_sha256")
    payload["evaluation_identity"]["content_sha256"] = hash_json(unsigned)
    paths["historical"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction identity differs"):
        module.materialize_two_step_manifest(
            legacy_evaluation_manifest=paths["legacy"],
            evidence_import_manifest=paths["imports"],
            historical_evaluation=paths["historical"],
            size_bins=paths["sizes"],
            output_path=tmp_path / "out.json",
            dev2_scene_ids=SCENES[:2],
            dev8_scene_ids=SCENES,
        )


def test_materialize_refuses_to_overwrite_a_frozen_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, relative=False)
    with pytest.raises(ValueError, match="overlaps an input file"):
        module.materialize_two_step_manifest(
            legacy_evaluation_manifest=paths["legacy"],
            evidence_import_manifest=paths["imports"],
            historical_evaluation=paths["historical"],
            size_bins=paths["sizes"],
            output_path=paths["legacy"],
            dev2_scene_ids=SCENES[:2],
            dev8_scene_ids=SCENES,
        )
