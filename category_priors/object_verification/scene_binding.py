"""Offline conversion to strict formal scene inputs; no model calls.

Historical args and mixed audit reports may be read ONLY by this conversion.
The formal loader consumes its two whitelist-only outputs, never those sources.
"""
from pathlib import Path
import json
import math

from .artifacts import file_digest, write_once

SPEC_PATH_FIELDS = {"point_cloud_path", "candidate_bank", "b0_output", "reservoir", "priors", "sparse_path", "images_path"}
SPEC_FIELDS = SPEC_PATH_FIELDS | {"schema", "kind", "scene_id", "sh_degree", "allow_principle_point_shift", "provenance"}
ASSET_KEYS = {"candidate_bank", "b0_output", "point_cloud_path", "contrastive_feature_point_cloud_path", "priors",
              "sam_checkpoint_path", "groundingdino_checkpoint_path", "groundingdino_config_path"}
SCENE_FIELDS = {"scene_id", "status", "checks", "point_count", "assets", "reservoir", "bank_xyz_sha256",
                "scene_scale_m_per_unit", "class_names", "saga20_names", "cameras", "rgb"}
INVENTORY_FIELDS = {"schema", "kind", "scene_id", "formal_scene_spec", "scene", "provenance"}
CHECKS = {"files_match_registered", "point_counts_match", "bank_xyz_order_matches", "feature_xyz_order_matches",
          "reservoir_b0_exactly_registered", "b0_current_ids_exactly_declared", "all_support_equals_parent_full",
          "binding_b0_hash_matches", "binding_bank_hash_matches"}


def validate_formal_binding(spec, inventory):
    if (set(spec) != SPEC_FIELDS or spec.get("schema") != "saga-object-formal-scene-spec-v1"
            or spec.get("kind") != "formal_runtime_scene"):
        raise ValueError("formal runtime rejects non-whitelisted scene fields, including old args, GT and manual inputs")
    if (set(inventory) != INVENTORY_FIELDS or inventory.get("schema") != "saga-object-formal-scene-assets-v1"
            or inventory.get("kind") != "formal_runtime_assets"):
        raise ValueError("formal runtime rejects mixed/offline asset inventory")
    if spec["scene_id"] not in {"scene0645_00", "scene0025_01"} or inventory["scene_id"] != spec["scene_id"]:
        raise ValueError("formal DEV2 scene identity mismatch")
    row = inventory["scene"]
    if (set(row) != SCENE_FIELDS or row["scene_id"] != spec["scene_id"]
            or set(row["assets"]) != ASSET_KEYS or set(row["checks"]) != CHECKS):
        raise ValueError("formal scene inventory contains undeclared/offline fields")
    if any(type(v) is not bool for v in row["checks"].values()):
        raise ValueError("asset checks must be booleans, never truthy text")
    if (type(row["point_count"]) is not int or row["point_count"] <= 0 or type(spec["sh_degree"]) is not int
            or spec["sh_degree"] < 0 or type(spec["allow_principle_point_shift"]) is not bool
            or type(row["scene_scale_m_per_unit"]) not in (int, float)
            or not math.isfinite(row["scene_scale_m_per_unit"]) or row["scene_scale_m_per_unit"] <= 0):
        raise ValueError("formal scene numeric/boolean contract invalid")
    for key, count in (("class_names", 32), ("saga20_names", 20)):
        names = row[key]
        if not isinstance(names, list) or len(names) != count or any(not isinstance(x, str) or not x.strip() for x in names) or len(set(names)) != count:
            raise ValueError("formal taxonomy must contain unique nonempty class strings")
    if not set(row["saga20_names"]) <= set(row["class_names"]):
        raise ValueError("SAGA20 must be a subset of the frozen 32 classes")
    def path_value(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("formal asset paths/provenance must be strings")
    def hash_value(value):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("formal SHA256 must be lowercase hexadecimal")
    for key in SPEC_PATH_FIELDS:
        path_value(spec[key])
    hash_value(row["bank_xyz_sha256"])
    if set(inventory["formal_scene_spec"]) != {"path", "sha256"}:
        raise ValueError("formal spec requires exact path/hash binding")
    path_value(inventory["formal_scene_spec"]["path"]); hash_value(inventory["formal_scene_spec"]["sha256"])
    for asset in row["assets"].values():
        if set(asset) != {"path", "exists", "sha256", "bytes", "registered_sha256", "matches_registered"}:
            raise ValueError("asset identity cannot contain GT/manual or undeclared metadata")
        path_value(asset["path"]); hash_value(asset["sha256"]); hash_value(asset["registered_sha256"])
        if type(asset["exists"]) is not bool or type(asset["matches_registered"]) is not bool or type(asset["bytes"]) is not int or asset["bytes"] < 0:
            raise ValueError("invalid formal file identity values")
    for domain in ("reservoir", "cameras", "rgb"):
        tree = row[domain]
        if set(tree) != {"path", "exists", "hash_algorithm", "sha256", "files"}:
            raise ValueError("undeclared formal asset directory fields")
        path_value(tree["path"]); hash_value(tree["sha256"])
        if type(tree["exists"]) is not bool or not isinstance(tree["files"], list) or tree["hash_algorithm"] != "ordered-relative-path-sha256-size-json-v1":
            raise ValueError("invalid formal directory identity")
        for asset in tree["files"]:
            if set(asset) != {"relative_path", "path", "exists", "sha256", "bytes"}:
                raise ValueError("directory file identity cannot contain offline fields")
            path_value(asset["path"]); path_value(asset["relative_path"]); hash_value(asset["sha256"])
            if type(asset["exists"]) is not bool or type(asset["bytes"]) is not int or asset["bytes"] < 0:
                raise ValueError("invalid formal directory file identity")
    for provenance in (spec["provenance"], inventory["provenance"]):
        if set(provenance) != {"conversion", "source_args_path", "source_args_sha256", "source_audit_path", "source_audit_sha256"}:
            raise ValueError("provenance is identity-only; no source document contents are allowed")
        if provenance["conversion"] != "offline_whitelist_conversion_v1":
            raise ValueError("unknown scene-binding conversion")
        for key in ("source_args_path", "source_audit_path"):
            path_value(provenance[key])
        for key in ("source_args_sha256", "source_audit_sha256"):
            hash_value(provenance[key])
    return row


def convert_scene_binding(args_path, audit_path, output_dir):
    """Read legacy sources offline and emit immutable formal allowlist files."""
    args_path, audit_path, output = Path(args_path), Path(audit_path), Path(output_dir)
    a = json.loads(args_path.read_text()); audit = json.loads(audit_path.read_text())
    sid = a["scene_id"]
    if sid not in {"scene0645_00", "scene0025_01"}:
        raise ValueError("DEV2 only")
    record = next(r for r in audit["scenes"] if r["scene_id"] == sid)
    if record["args"]["path"] != str(args_path) or record["args"]["sha256"] != file_digest(args_path):
        raise ValueError("legacy binding differs from measured audit")
    if record["status"] != "complete" or set(record["checks"]) != CHECKS or not all(record["checks"].values()):
        raise ValueError("unresolved source asset identities")
    provenance = {"conversion": "offline_whitelist_conversion_v1", "source_args_path": str(args_path),
                  "source_args_sha256": file_digest(args_path), "source_audit_path": str(audit_path),
                  "source_audit_sha256": file_digest(audit_path)}
    spec = {"schema": "saga-object-formal-scene-spec-v1", "kind": "formal_runtime_scene", "scene_id": sid,
            "sh_degree": a["sh_degree"], "allow_principle_point_shift": a["allow_principle_point_shift"],
            "provenance": provenance, **{key: a[key] for key in SPEC_PATH_FIELDS}}
    spec_path = output / f"{sid}_scene_spec.json"
    spec_sha = write_once(spec_path, spec)
    scene = {key: record[key] for key in SCENE_FIELDS}
    scene["assets"] = {key: record["assets"][key] for key in ASSET_KEYS}
    inventory = {"schema": "saga-object-formal-scene-assets-v1", "kind": "formal_runtime_assets",
                 "scene_id": sid, "formal_scene_spec": {"path": str(spec_path), "sha256": spec_sha},
                 "scene": scene, "provenance": provenance}
    validate_formal_binding(spec, inventory)
    inventory_path = output / f"{sid}_scene_assets.json"
    write_once(inventory_path, inventory)
    return spec_path, inventory_path
