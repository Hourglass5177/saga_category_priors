import copy
import json

import pytest

from category_priors.object_verification.artifacts import file_digest
from category_priors.object_verification.scene_binding import (ASSET_KEYS, CHECKS, SPEC_PATH_FIELDS,
    convert_scene_binding, validate_formal_binding)
from category_priors.object_verification.frozen_scene import FrozenSceneAssets


def source_fixture(tmp_path):
    args = tmp_path / "historical_args.json"
    args.write_text(json.dumps({"scene_id": "scene0645_00", "sh_degree": 0,
        "allow_principle_point_shift": False, **{key: f"/cloud/{key}" for key in SPEC_PATH_FIELDS},
        "labels_path": "/old/labels", "masks_path": "/old/masks", "gt_path": "/offline/gt"}))
    asset = {"path": "/cloud/file", "exists": True, "sha256": "a" * 64, "bytes": 1,
             "registered_sha256": "a" * 64, "matches_registered": True}
    tree = {"path": "/cloud/dir", "exists": True, "hash_algorithm": "ordered-relative-path-sha256-size-json-v1", "sha256": "b" * 64, "files": []}
    scene = {"scene_id": "scene0645_00", "status": "complete", "checks": {k: True for k in CHECKS},
        "point_count": 3, "assets": {key: asset for key in ASSET_KEYS}, "reservoir": tree,
        "bank_xyz_sha256": "c" * 64, "scene_scale_m_per_unit": 1.0, "class_names": [f"class{i}" for i in range(32)], "saga20_names": [f"class{i}" for i in range(20)],
        "cameras": tree, "rgb": tree, "args": {"path": str(args), "sha256": file_digest(args)},
        "candidate_rows": [{"historical_reachable_field_ignored": False}]}
    audit = tmp_path / "mixed_audit.json"
    audit.write_text(json.dumps({"scenes": [scene], "controls": {"objects": [{"gt_key": "manual-gt"}]}}))
    return args, audit


def test_offline_conversion_removes_old_fields_and_does_not_embed_control_data(tmp_path):
    args, audit = source_fixture(tmp_path)
    sp, ap = convert_scene_binding(args, audit, tmp_path / "formal")
    spec, assets = json.loads(sp.read_text()), json.loads(ap.read_text())
    validate_formal_binding(spec, assets)
    combined = sp.read_text() + ap.read_text()
    for key in ("gt_key", "manual-gt", "labels_path", "masks_path", "candidate_rows", "controls"):
        assert key not in combined
    assert assets["formal_scene_spec"]["sha256"] == file_digest(sp)
    assert convert_scene_binding(args, audit, tmp_path / "formal") == (sp, ap)


@pytest.mark.parametrize("field", ["gt_path", "manual_controls", "labels_path", "alpha_cache_dir", "output_dir"])
def test_formal_loader_rejects_old_or_offline_fields_before_scene_loading(tmp_path, field):
    args, audit = source_fixture(tmp_path)
    sp, ap = convert_scene_binding(args, audit, tmp_path / "formal")
    spec = json.loads(sp.read_text()); spec[field] = "/forbidden"
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="non-whitelisted"):
        FrozenSceneAssets.load(bad, ap)


def test_formal_inventory_rejects_mixed_and_nested_manual_fields(tmp_path):
    args, audit = source_fixture(tmp_path)
    sp, ap = convert_scene_binding(args, audit, tmp_path / "formal")
    spec, assets = json.loads(sp.read_text()), json.loads(ap.read_text())
    extra = copy.deepcopy(assets); extra["controls"] = []
    with pytest.raises(ValueError, match="mixed/offline"):
        validate_formal_binding(spec, extra)
    extra = copy.deepcopy(assets); extra["scene"]["assets"]["candidate_bank"]["gt_path"] = "forbidden"
    with pytest.raises(ValueError, match="GT/manual"):
        validate_formal_binding(spec, extra)


@pytest.mark.parametrize("key,value", [("class_names", [{"gt": "payload"}] * 32), ("scene_scale_m_per_unit", float("nan")), ("point_count", True)])
def test_allowed_fields_cannot_hide_untyped_payloads(tmp_path, key, value):
    args, audit = source_fixture(tmp_path)
    sp, ap = convert_scene_binding(args, audit, tmp_path / "formal")
    spec, assets = json.loads(sp.read_text()), json.loads(ap.read_text())
    assets["scene"][key] = value
    with pytest.raises(ValueError):
        validate_formal_binding(spec, assets)


def test_truthy_string_check_is_rejected(tmp_path):
    args, audit = source_fixture(tmp_path)
    sp, ap = convert_scene_binding(args, audit, tmp_path / "formal")
    spec, assets = json.loads(sp.read_text()), json.loads(ap.read_text())
    assets["scene"]["checks"]["files_match_registered"] = "false"
    with pytest.raises(ValueError, match="booleans"):
        validate_formal_binding(spec, assets)
