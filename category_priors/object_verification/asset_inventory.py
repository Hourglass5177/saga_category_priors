"""Read-only CPU inventory of frozen assets; never imports an old experiment runner.

This report certifies bytes and index alignment only. It is not a supply,
identity, model-runtime, or mechanism acceptance report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np


def file_identity(path):
    p = Path(path)
    row = {"path": str(p), "exists": p.is_file()}
    if p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        row.update(sha256=h.hexdigest(), bytes=p.stat().st_size)
    return row


def tree_identity(path):
    p = Path(path)
    if not p.is_dir():
        return {"path": str(p), "exists": False}
    rows = []
    for q in sorted(p.rglob("*")):
        if q.is_file():
            rows.append({"relative_path": q.relative_to(p).as_posix(), **file_identity(q)})
    # This explicit algorithm is new and is not compared to undocumented old tree hashes.
    encoded = json.dumps([(r["relative_path"], r["sha256"], r["bytes"]) for r in rows], separators=(",", ":")).encode()
    return {"path": str(p), "exists": True, "hash_algorithm": "ordered-relative-path-sha256-size-json-v1",
            "sha256": hashlib.sha256(encoded).hexdigest(), "files": rows}


def xyz_hash(xyz):
    xyz = np.ascontiguousarray(xyz, dtype="<f8")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("finite N x 3 XYZ required")
    return hashlib.sha256(np.asarray(xyz.shape, dtype="<i8").tobytes() + xyz.tobytes()).hexdigest()


def validate_csr(indptr, ids, rows, point_count):
    indptr, ids = np.asarray(indptr), np.asarray(ids)
    if indptr.dtype.kind not in "iu" or ids.dtype.kind not in "iu":
        raise ValueError("integer CSR indices required")
    if indptr.shape != (rows + 1,) or ids.ndim != 1 or indptr[0] != 0 or indptr[-1] != len(ids) or np.any(np.diff(indptr) < 0):
        raise ValueError("invalid CSR offsets")
    if np.any(ids < 0) or np.any(ids >= point_count):
        raise ValueError("CSR Gaussian index outside full frozen point array")
    for a, b in zip(indptr[:-1], indptr[1:]):
        if len(np.unique(ids[a:b])) != b - a:
            raise ValueError("duplicate Gaussian within one CSR row")
    return {"rows": rows, "entries": len(ids), "empty_rows": int(np.sum(np.diff(indptr) == 0))}


def audit_scene(args_path):
    from plyfile import PlyData
    args_path = Path(args_path)
    a = json.loads(args_path.read_text())
    old = json.loads(Path(a["asset_manifest"]).read_text())
    expected = {x["path"]: x.get("sha256") for x in old["artifacts"]}
    files = {}
    for key in ("candidate_bank", "b0_output", "point_cloud_path", "contrastive_feature_point_cloud_path",
                "priors", "sam_checkpoint_path", "groundingdino_checkpoint_path", "groundingdino_config_path"):
        row = file_identity(a[key]); row["registered_sha256"] = expected.get(a[key])
        row["matches_registered"] = bool(row.get("sha256") and row["sha256"] == row["registered_sha256"])
        files[key] = row
    p = PlyData.read(a["point_cloud_path"])["vertex"]
    xyz = np.column_stack([p[k] for k in ("x", "y", "z")]); n = len(xyz)
    feature = PlyData.read(a["contrastive_feature_point_cloud_path"])["vertex"]
    feature_xyz = np.column_stack([feature[k] for k in ("x", "y", "z")])
    with np.load(a["candidate_bank"], allow_pickle=False) as z:
        schema = {k: {"dtype": str(z[k].dtype), "shape": list(z[k].shape)} for k in z.files}
        m = json.loads(z["metadata_json"].item())
        full, core = z["branch_full_labels"].copy(), z["branch_core_labels"].copy()
    b = json.loads(Path(a["b0_output"]).read_text()); labels = np.asarray(b["point_labels"], dtype=np.int64)
    rp = Path(a["reservoir"])
    rm = json.loads((rp / "reservoir.json").read_text()); binding = json.loads((rp / "binding.json").read_text())
    candidates = []
    with np.load(rp / "reservoir.npz", allow_pickle=False) as z:
        rs = {k: {"dtype": str(z[k].dtype), "shape": list(z[k].shape)} for k in z.files}
        csrs = {prefix: validate_csr(z[prefix + "_indptr"], z[prefix + "_ids"], len(rm["candidates"]), n) for prefix in ("support", "anchor")}
        b0_equal = np.array_equal(z["b0_labels"], labels)
        for i, row in enumerate(rm["candidates"]):
            support = z["support_ids"][z["support_indptr"][i]:z["support_indptr"][i + 1]]
            anchor = z["anchor_ids"][z["anchor_indptr"][i]:z["anchor_indptr"][i + 1]]
            parents = row["parent_candidate_ids"]
            original = np.flatnonzero(np.isin(full, parents)); original_core = np.flatnonzero(np.isin(core, parents))
            candidates.append({"row": i, "candidate_id": row["candidate_id"], "parent_candidate_ids": parents,
                "branch_class": row["branch_class"], "anchor_stage": row["anchor_stage"],
                "support_count": len(support), "anchor_count": len(anchor), "original_core_count": len(original_core),
                "support_equals_parent_full": bool(np.array_equal(np.sort(support), original)),
                "anchor_outside_support_count": len(np.setdiff1d(anchor, support)),
                "core_outside_support_count": len(np.setdiff1d(original_core, support)),
                "historical_reachable_field_ignored": row.get("reachable")})
    current_ids = sorted(int(x) for x in np.unique(labels) if x >= 0)
    declared_ids = sorted(int(x) for x in b["instances"])
    checks = {"files_match_registered": all(x["matches_registered"] for x in files.values()),
              "point_counts_match": n == len(labels) == m["point_count"] == rm["point_count"] == len(feature_xyz),
              "bank_xyz_order_matches": xyz_hash(xyz) == m["gaussian_xyz_sha256"],
              "feature_xyz_order_matches": np.array_equal(xyz, feature_xyz),
              "reservoir_b0_exactly_registered": b0_equal,
              "b0_current_ids_exactly_declared": current_ids == declared_ids,
              "all_support_equals_parent_full": all(c["support_equals_parent_full"] for c in candidates),
              "binding_b0_hash_matches": binding["registered_b0_sha256"] == files["b0_output"]["sha256"],
              "binding_bank_hash_matches": binding["candidate_bank_sha256"] == files["candidate_bank"]["sha256"]}
    return {"scene_id": a["scene_id"], "args": file_identity(args_path), "assets": files,
            "checks": checks, "status": "complete" if all(checks.values()) else "failed",
            "point_count": n, "bank_schema": schema, "bank_metadata_fields": sorted(m),
            "bank_xyz_sha256": m["gaussian_xyz_sha256"], "scene_scale_m_per_unit": m["scene_scale_m_per_unit"],
            "class_names": m["class_names"], "saga20_names": m["saga20_names"],
            "reservoir": tree_identity(rp), "reservoir_array_schema": rs, "csr_checks": csrs,
            "candidate_rows": candidates, "b0_instance_count": len(declared_ids),
            "b0_background_count": int(np.sum(labels < 0)), "b0_instances": b["instances"],
            "historical_prediction_contract": b.get("prediction_contract"),
            "historical_contract_note": "orphan fields describe prior normalization; current point_labels and instances are checked directly",
            "cameras": tree_identity(a["sparse_path"]), "rgb": tree_identity(a["images_path"])}


def audit_controls(manual_path):
    p = Path(manual_path); m = json.loads(p.read_text())
    validation = json.loads((p.parent / "annotation_validation.json").read_text())
    by_key = {(r["scene_id"], r["candidate_id"], r["camera_index"]): r for r in validation["validated_views"]}
    objects = []
    for obj in m["objects"]:
        views = []
        for v in obj["views"]:
            obs = v["observation"]; key = (obj["scene_id"], obj["candidate_id"], obs["camera_index"])
            record = by_key[key]; assets = {}
            for kind in ("rgb", "foreground", "uncertain"):
                row = file_identity(record["paths"][kind]); row["expected_sha256"] = v[kind + "_sha256"]
                row["matches_registered"] = row.get("sha256") == row["expected_sha256"]
                assets[kind] = row
            views.append({"camera_index": obs["camera_index"], "image_name": obs["image_name"],
                          "observation": obs, "same_object_confirmed": v.get("same_object_confirmed"),
                          "annotation_origin": v.get("annotation_origin"), "assets": assets})
        objects.append({k: obj[k] for k in ("scene_id", "candidate_id", "target_id", "target_class", "gt_key")} | {"views": views})
    return {"role": "offline_diagnostic_only", "manual_manifest": file_identity(p),
            "validation_manifest": file_identity(p.parent / "annotation_validation.json"),
            "objects": objects, "all_referenced_bytes_match": all(a["matches_registered"] for o in objects for v in o["views"] for a in v["assets"].values()),
            "new_identity_mechanism_acceptance": "pending_not_established_by_historical_annotations"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args", action="append", required=True)
    parser.add_argument("--manual-controls", required=True)
    parser.add_argument("--output", required=True)
    a = parser.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    scenes = [audit_scene(p) for p in a.args]
    controls = audit_controls(a.manual_controls)
    result = {"schema": "saga-object-verification-asset-inventory-v1", "python": platform.python_version(),
              "numpy": np.__version__, "gpu_calls": 0, "scenes": scenes, "controls": controls,
              "scientific_validity": "not_tested", "supply_audit": "pending",
              "status": "complete" if all(s["status"] == "complete" for s in scenes) and controls["all_referenced_bytes_match"] else "failed"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(out), "status": result["status"], "scenes": [{"scene": s["scene_id"], "checks": s["checks"]} for s in scenes]}))


if __name__ == "__main__":
    main()
