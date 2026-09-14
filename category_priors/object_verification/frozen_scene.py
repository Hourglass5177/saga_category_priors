"""Actual frozen DEV2 scene access without a retired experiment runtime.

Asset loading and camera geometry are CPU-only. NativeMeasurementRenderer is
explicitly constructed inside a budgeted GPU task, never during module import.
GT, manual masks, old semantic masks, graph and KNN are not loaded here.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from .artifacts import digest, file_digest
from .asset_inventory import validate_csr, xyz_hash
from .contracts import CandidateSource
from .transactions import SceneObject, SceneSnapshot, make_scene


def quaternion_rotation(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(q @ q, 1.0, atol=1e-6):
        raise ValueError("COLMAP quaternion must be finite and unit length")
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def infer_sh_degree(property_names):
    names = set(property_names)
    dc = {n for n in names if n.startswith("f_dc_")}
    rest = {n for n in names if n.startswith("f_rest_")}
    if dc != {f"f_dc_{i}" for i in range(3)} or rest != {f"f_rest_{i}" for i in range(len(rest))}:
        raise ValueError("Gaussian SH properties must be contiguous exact RGB coefficients")
    root = math.isqrt(1 + len(rest) // 3)
    if len(rest) % 3 or 3 * root * root - 3 != len(rest):
        raise ValueError("Gaussian SH property count is not a complete degree")
    return root - 1


@dataclass(frozen=True)
class CameraSpec:
    uid: str
    old_index: int
    image_name: str
    image_path: str
    image_sha256: str
    width: int
    height: int
    fx: float
    fy: float
    colmap_principal_xy: tuple[float, float]
    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]
    scale_m_per_unit: float
    source_sha256: str

    def __post_init__(self):
        r, t = np.asarray(self.rotation), np.asarray(self.translation)
        numbers = np.asarray((self.fx, self.fy, self.scale_m_per_unit, *self.colmap_principal_xy), dtype=np.float64)
        if (r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all()
                or not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1.0, atol=1e-6)
                or not np.isfinite(numbers).all() or len(self.colmap_principal_xy) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (self.width, self.height, self.old_index))
                or min(self.width, self.height, self.fx, self.fy, self.scale_m_per_unit) <= 0
                or not all((self.uid, self.image_sha256, self.source_sha256))):
            raise ValueError("invalid frozen camera geometry")

    @property
    def effective_principal_xy(self):
        # Native splat NDC-to-pixel maps zero to (size - 1) / 2. Keep the
        # historical centered projection, not COLMAP's shifted principal point.
        return ((self.width - 1) / 2, (self.height - 1) / 2)

    @property
    def center_scene(self):
        return -np.asarray(self.rotation).T @ np.asarray(self.translation)

    @property
    def center_m(self):
        return self.center_scene * self.scale_m_per_unit

    def optical_z_m(self, xyz_scene):
        return (np.asarray(xyz_scene) @ np.asarray(self.rotation).T + self.translation)[..., 2] * self.scale_m_per_unit

    def project(self, xyz_scene):
        value = np.asarray(xyz_scene) @ np.asarray(self.rotation).T + self.translation
        z = value[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = value[..., :2] / z[..., None] * [self.fx, self.fy] + self.effective_principal_xy
        return xy, z * self.scale_m_per_unit


def load_text_cameras(sparse_path, images_path, *, scene_id, scale_m_per_unit, registered_files):
    """Read the actual frozen text COLMAP format; unsupported files fail closed."""
    from PIL import Image
    sparse, images = Path(sparse_path), Path(images_path)
    def check(path):
        path = Path(path)
        if str(path) not in registered_files or file_digest(path) != registered_files[str(path)]:
            raise ValueError(f"unregistered or changed camera/RGB asset: {path}")
    cp, ip = sparse / "cameras.txt", sparse / "images.txt"
    check(cp); check(ip)
    intrinsics = {}
    for line in cp.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = line.split()
        if len(row) != 8 or row[1] != "PINHOLE":
            raise ValueError("current frozen camera loader requires registered PINHOLE text")
        intrinsics[int(row[0])] = (int(row[2]), int(row[3]), *map(float, row[4:]))
    extrinsics = []
    with ip.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = line.split()
            if len(row) != 10:
                raise ValueError("COLMAP image header must have ten fields")
            q, t = tuple(map(float, row[1:5])), tuple(map(float, row[5:8]))
            extrinsics.append((row[9], int(row[8]), quaternion_rotation(q), t))
            # Consume the points2D line, even when blank; never parse it as pose.
            if f.readline() == "":
                raise ValueError("missing COLMAP points2D record")
    result = []
    source = digest({"cameras.txt": registered_files[str(cp)], "images.txt": registered_files[str(ip)],
                     "principal_rule": "historical_centered_ndc_pixel_size_minus_one_over_two"})
    for index, (name, cid, r, t) in enumerate(sorted(extrinsics, key=lambda x: Path(x[0]).stem)):
        path = images / Path(name).name
        if not path.is_file():
            candidates = sorted(q for q in images.iterdir() if q.is_file() and q.stem == Path(name).stem)
            if len(candidates) != 1:
                raise ValueError("missing or ambiguous exact-stem RGB")
            path = candidates[0]
        check(path)
        with Image.open(path) as image:
            width, height = image.size
        ow, oh, fx, fy, cx, cy = intrinsics[cid]
        image_name = Path(name).stem
        result.append(CameraSpec(f"{scene_id}:{image_name}", index, image_name, str(path),
            registered_files[str(path)], width, height, fx * width / ow, fy * height / oh,
            (cx * width / ow, cy * height / oh), tuple(map(tuple, r)), t, scale_m_per_unit, source))
    if len({c.uid for c in result}) != len(result) or not result:
        raise ValueError("duplicate or empty camera registry")
    return tuple(result)


@dataclass
class FrozenSceneAssets:
    scene_id: str
    xyz_scene: np.ndarray
    scale_m_per_unit: float
    cameras: tuple[CameraSpec, ...]
    sources: tuple[CandidateSource, ...]
    b0_scene: SceneSnapshot
    b0_metadata: Mapping
    priors: Mapping
    point_cloud_path: str
    classes32: tuple[str, ...]
    saga20: tuple[str, ...]
    identity_sha256: str
    old_candidate_ids: Mapping[str, int]
    old_b0_ids: Mapping[str, int]
    prior_sha256: str
    sh_degree: int = 0

    @classmethod
    def load(cls, args_path, inventory_path):
        from .scene_binding import validate_formal_binding
        args_path = Path(args_path)
        a = json.loads(args_path.read_text()); inventory = json.loads(Path(inventory_path).read_text())
        record = validate_formal_binding(a, inventory)
        from plyfile import PlyData
        if record["status"] != "complete" or not all(record["checks"].values()):
            raise ValueError("asset inventory has unresolved identity checks")
        if inventory["formal_scene_spec"]["path"] != str(args_path) or file_digest(args_path) != inventory["formal_scene_spec"]["sha256"]:
            raise ValueError("formal scene spec not bound to current asset inventory")
        if a.get("allow_principle_point_shift") is not False:
            raise ValueError("freeze historical centered-principal camera convention")
        # Only whitelist formal paths are read. Provenance hashes identify the
        # offline conversion, but those historical source files are never opened.
        used = ("candidate_bank", "b0_output", "point_cloud_path", "priors")
        for key in used:
            row = record["assets"][key]
            if a[key] != row["path"] or file_digest(a[key]) != row["sha256"]:
                raise ValueError(f"actual input changed: {key}")
        for row in record["reservoir"]["files"]:
            if file_digest(row["path"]) != row["sha256"]:
                raise ValueError("reservoir changed after audit")
        vertices = PlyData.read(a["point_cloud_path"])["vertex"]
        sh_degree = infer_sh_degree(vertices.data.dtype.names)
        if sh_degree != a["sh_degree"]:
            raise ValueError("actual PLY SH degree differs from registered args")
        xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")])
        if xyz_hash(xyz) != record["bank_xyz_sha256"]:
            raise ValueError("Gaussian point index order differs from frozen CandidateBank")
        xyz = np.frombuffer(xyz.tobytes(), dtype=xyz.dtype).reshape(xyz.shape)
        n, sid = len(xyz), a["scene_id"]
        rp = Path(a["reservoir"])
        if str(rp) != record["reservoir"]["path"]:
            raise ValueError("runtime reservoir differs from inventory")
        rm = json.loads((rp / "reservoir.json").read_text())
        sources, old_candidates = [], {}
        with np.load(rp / "reservoir.npz", allow_pickle=False) as z:
            for prefix in ("anchor", "support"):
                validate_csr(z[prefix+"_indptr"], z[prefix+"_ids"], len(rm["candidates"]), n)
            for i, row in enumerate(rm["candidates"]):
                ids = {}
                for prefix in ("anchor", "support"):
                    ids[prefix] = tuple(map(int, z[prefix+"_ids"][z[prefix+"_indptr"][i]:z[prefix+"_indptr"][i+1]]))
                uid = f"{sid}:C0:{int(row['candidate_id']):06d}"
                source_hash = digest({"reservoir": record["reservoir"]["sha256"], "row": i, "metadata": row,
                                      "ancestry_definition": "historical_3d_bank_no_object_view_claim"})
                sources.append(CandidateSource(uid, tuple(f"{sid}:parent:{int(p):06d}" for p in row["parent_candidate_ids"]),
                                               ids["anchor"], ids["support"], (), source_hash))
                old_candidates[uid] = row["candidate_id"]
            reservoir_b0 = z["b0_labels"].copy()
        b0 = json.loads(Path(a["b0_output"]).read_text()); labels = np.asarray(b0["point_labels"])
        if not np.array_equal(labels, reservoir_b0):
            raise ValueError("bound B0 labels no longer match original")
        objects, old_b0, metadata = [], {}, {}
        for key, row in sorted(b0["instances"].items(), key=lambda pair: int(pair[0])):
            uid = f"{sid}:B0:{int(key):06d}"
            objects.append(SceneObject(uid, digest({"b0": record["assets"]["b0_output"]["sha256"], "id": key}),
                                       tuple(map(int, np.flatnonzero(labels == int(key)))), row["class"], row["score"]))
            old_b0[uid], metadata[uid] = int(key), row
        scale = record["scene_scale_m_per_unit"]
        registered = {r["path"]: r["sha256"] for domain in ("cameras", "rgb") for r in record[domain]["files"]}
        cameras = load_text_cameras(a["sparse_path"], a["images_path"], scene_id=sid, scale_m_per_unit=scale, registered_files=registered)
        identity = digest({"scene": sid, "point_cloud": record["assets"]["point_cloud_path"]["sha256"],
                           "reservoir": record["reservoir"]["sha256"], "b0": record["assets"]["b0_output"]["sha256"],
                           "camera_registry": cameras, "prior": record["assets"]["priors"]["sha256"]})
        scene = make_scene(sid, xyz * scale, objects)
        return cls(sid, xyz, scale, cameras, tuple(sources), scene, metadata,
                   json.loads(Path(a["priors"]).read_text()), a["point_cloud_path"], tuple(record["class_names"]),
                   tuple(record["saga20_names"]), identity, old_candidates, old_b0, record["assets"]["priors"]["sha256"], sh_degree)


class NativeMeasurementRenderer:
    """Native reference-only measurements. Construction allocates CUDA memory."""
    def __init__(self, assets: FrozenSceneAssets):
        import torch
        from scene.gaussian_model import GaussianModel
        from .measurement import loaded_backend_identity
        self.gaussians = GaussianModel(assets.sh_degree)
        self.gaussians.load_ply(assets.point_cloud_path)
        if not np.array_equal(self.gaussians.get_xyz.detach().cpu().numpy(), assets.xyz_scene):
            raise ValueError("native Gaussian loader changed point order")
        for value in vars(self.gaussians).values():
            if isinstance(value, torch.Tensor) and value.is_leaf:
                value.requires_grad_(False)
        self.pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        self.background = torch.zeros(3, device="cuda")
        self.identity = loaded_backend_identity()
        self.identity["scene_input_sha256"] = assets.identity_sha256
        self._cameras = OrderedDict()

    def _camera(self, camera, rgb=None):
        import torch
        from PIL import Image
        from scene.cameras import Camera
        if camera.uid not in self._cameras:
            if rgb is None:
                if file_digest(camera.image_path) != camera.image_sha256:
                    raise ValueError("native camera RGB changed")
                with Image.open(camera.image_path) as image:
                    rgb = np.array(image.convert("RGB"))
            tensor = torch.tensor(rgb.transpose(2, 0, 1).copy(), dtype=torch.float32) / 255
            native = Camera(colmap_id=camera.old_index, R=np.asarray(camera.rotation).T,
                T=np.asarray(camera.translation), FoVx=2*math.atan(camera.width/(2*camera.fx)),
                FoVy=2*math.atan(camera.height/(2*camera.fy)), image=tensor, gt_alpha_mask=None,
                image_name=camera.image_name, uid=camera.old_index, cx=None, cy=None)
            if not np.allclose(native.camera_center.detach().cpu().numpy(), camera.center_scene, atol=1e-5):
                raise ValueError("native camera center disagrees with frozen COLMAP convention")
            self._cameras[camera.uid] = native
        self._cameras.move_to_end(camera.uid)
        while len(self._cameras) > 2:
            self._cameras.popitem(last=False)
        return self._cameras[camera.uid]

    def contributors(self, camera, rgb):
        import torch
        from .measurement import render_contributors
        with torch.no_grad():
            return render_contributors(self._camera(camera, rgb), self.gaussians, self.pipeline, self.background)

    def alpha(self, camera, masks, valid):
        from .measurement import render_reference
        return render_reference(self._camera(camera), self.gaussians, self.pipeline, self.background,
                                masks, valid_pixels=valid)
