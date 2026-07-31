import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

try:
    import supervision as sv
except Exception:
    sv = None

try:
    from groundingdino.util.inference import Model
except Exception:
    Model = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SHAPER_DIR = REPO_ROOT / "ShapeR"
if str(SHAPER_DIR) not in sys.path:
    sys.path.insert(0, str(SHAPER_DIR))

from split_point_cloud_by_class import read_binary_vertex_ply, write_binary_vertex_ply
from utils.read_write_model import read_cameras_binary, read_images_binary


SINGLE_MODEL_FILTED_PLY_NAME = "single_model_filted.ply"

DEFAULT_CLASSES = [
    "chair",
    "table",
    "plant",
    "flower",
    "foliage",
    "tv",
    "painting",
    "sofa",
    "cabinet",
    "bed",
    "wall",
    "floor",
    "ceiling",
    "person",
    "socket",
    "book",
    "remote",
    "key",
    "lamp",
    "speaker",
    "computer",
    "fan",
    "refrigerator",
    "robot",
    "cup",
    "vase",
    "phone",
    "trash can",
]


def words_to_tensors(word_list, dim=32, device="cpu"):
    num_classes = len(word_list)
    if dim >= num_classes - 1:
        matrix = torch.eye(num_classes, device=device) - (1.0 / num_classes)
        u, _, _ = torch.linalg.svd(matrix)
        feats = u[:, : num_classes - 1]
        pad_size = dim - (num_classes - 1)
        if pad_size > 0:
            feats = F.pad(feats, (0, pad_size), "constant", 0)
    else:
        words = sorted(word_list)
        seed = int(hashlib.md5("|".join(words).encode()).hexdigest()[:8], 16)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        feats = torch.randn(num_classes, dim, device=device, generator=generator)
        feats.requires_grad = True
        optimizer = torch.optim.Adam([feats], lr=0.1)
        for _ in range(200):
            optimizer.zero_grad()
            feats_norm = F.normalize(feats, p=2, dim=1)
            gram = torch.mm(feats_norm, feats_norm.t())
            target = torch.eye(num_classes, device=device)
            loss = (gram - target).pow(2).mean()
            loss.backward()
            optimizer.step()
        feats = feats.detach()
    return F.normalize(feats, p=2, dim=1)


def sanitize_filename(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    return s or "unknown"


def resolve_scene_paths(scene_dir: Path) -> dict[str, Path]:
    sparse_dir = scene_dir / "fastRecon" / "dense" / "sparse" / "0"
    return {
        "single_model_json": scene_dir / "single_model" / "single_model.json",
        "source_ply": scene_dir / "output_models" / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        "image_dir": sparse_dir / "images",
        "colmap_dir": sparse_dir,
    }


def resolve_weight_path(path_value: str | Path) -> str:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return str(path)
    candidates = [
        Path.cwd() / path,
        SCRIPT_DIR / path,
        REPO_ROOT / "saga" / path,
        REPO_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"未找到权重/配置文件: {path_value}")


def load_single_model_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 应为 JSON object")
    class_name = data.get("class") or data.get("class_name") or data.get("semantic")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError(f"{path} 缺少 class/class_name/semantic 字段")
    data["class"] = class_name.strip()
    bbox = np.asarray(data.get("bbox"), dtype=np.float64)
    if bbox.size != 24:
        raise ValueError(f"{path} 的 bbox 应包含 24 个数（8 个 3D 角点）")
    data["bbox"] = bbox.reshape(8, 3).tolist()
    return data


def caption_from_config(config: dict) -> str:
    for key in ("caption", "prompt", "text", "semantic", "semantics", "description"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"a {config['class']}"


def load_colmap_cameras(colmap_dir: Path) -> dict[str, dict]:
    cameras_bin = colmap_dir / "cameras.bin"
    images_bin = colmap_dir / "images.bin"
    if not cameras_bin.exists() or not images_bin.exists():
        raise FileNotFoundError(f"未找到 COLMAP 文件: {cameras_bin} 或 {images_bin}")
    cam_dict = read_cameras_binary(str(cameras_bin))
    img_dict = read_images_binary(str(images_bin))
    cameras = {}
    for _, img in img_dict.items():
        cam = cam_dict[img.camera_id]
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = img.qvec2rotmat()
        w2c[:3, 3] = img.tvec
        if cam.model in ("PINHOLE", "SIMPLE_PINHOLE"):
            fx, fy, cx, cy = cam.params[:4]
        else:
            fx = fy = cam.params[0]
            cx = cam.params[1]
            cy = cam.params[2]
        stem = Path(img.name).stem
        cameras[stem] = {
            "name": img.name,
            "stem": stem,
            "w2c": w2c,
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }
    return cameras


def resolve_image_path(image_dir: Path, image_name: str) -> Path | None:
    direct = image_dir / Path(image_name).name
    if direct.exists():
        return direct
    stem = Path(image_name).stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def select_points_in_oriented_bbox(points_xyz: np.ndarray, bbox_corners: np.ndarray, margin: float = 1e-4) -> np.ndarray:
    corners = np.asarray(bbox_corners, dtype=np.float64).reshape(8, 3)
    best = None
    for origin_idx, origin in enumerate(corners):
        dists = np.linalg.norm(corners - origin, axis=1)
        neighbor_idxs = [i for i in np.argsort(dists) if i != origin_idx][:3]
        if len(neighbor_idxs) < 3:
            continue
        basis = (corners[neighbor_idxs] - origin).T
        det = abs(float(np.linalg.det(basis)))
        if det < 1e-10:
            continue
        inv_basis = np.linalg.inv(basis)
        corner_coords = (corners - origin) @ inv_basis.T
        if not np.all((corner_coords >= -1e-3) & (corner_coords <= 1.0 + 1e-3)):
            continue
        score = np.abs(corner_coords * (1.0 - corner_coords)).sum() - det * 1e-6
        if best is None or score < best[0]:
            best = (score, origin, inv_basis)
    if best is None:
        return np.all((points_xyz >= corners.min(axis=0) - margin) & (points_xyz <= corners.max(axis=0) + margin), axis=1)
    _, origin, inv_basis = best
    coords = (points_xyz.astype(np.float64) - origin) @ inv_basis.T
    return np.all((coords >= -margin) & (coords <= 1.0 + margin), axis=1)


def select_points_from_json(vertices: np.ndarray, config: dict, selection_mode: str) -> tuple[np.ndarray, str]:
    points_xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    bbox_mask = select_points_in_oriented_bbox(points_xyz, np.asarray(config["bbox"], dtype=np.float64))
    indices = config.get("indices")
    index_mask = None
    if isinstance(indices, list) and indices:
        idx = np.asarray(indices, dtype=np.int64)
        valid = (idx >= 0) & (idx < len(vertices))
        if np.any(valid):
            index_mask = np.zeros(len(vertices), dtype=bool)
            index_mask[idx[valid]] = True
    if selection_mode == "indices":
        if index_mask is None:
            raise ValueError("single_model.json 中没有可用 indices")
        return index_mask, "indices"
    if selection_mode == "auto" and int(bbox_mask.sum()) == 0 and index_mask is not None:
        return index_mask, "indices"
    return bbox_mask, "bbox"


def project_points(points_xyz: np.ndarray, camera: dict, image_shape: tuple[int, int]):
    h, w = image_shape
    pts_h = np.concatenate([points_xyz, np.ones((len(points_xyz), 1), dtype=np.float64)], axis=1)
    cam_pts = (camera["w2c"] @ pts_h.T).T[:, :3]
    z = cam_pts[:, 2]
    valid = z > 1e-3
    ui = np.zeros(len(cam_pts), dtype=np.int32)
    vi = np.zeros(len(cam_pts), dtype=np.int32)
    if np.any(valid):
        u = camera["fx"] * cam_pts[valid, 0] / z[valid] + camera["cx"]
        v = camera["fy"] * cam_pts[valid, 1] / z[valid] + camera["cy"]
        ui[valid] = np.round(u).astype(np.int32)
        vi[valid] = np.round(v).astype(np.int32)
    in_bounds = valid & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    return np.stack([ui[in_bounds], vi[in_bounds]], axis=1), float(in_bounds.sum()) / max(int(valid.sum()), 1)


def project_points_with_depth(points_xyz: np.ndarray, camera: dict, image_shape: tuple[int, int]):
    h, w = image_shape
    pts_h = np.concatenate([points_xyz, np.ones((len(points_xyz), 1), dtype=np.float64)], axis=1)
    cam_pts = (camera["w2c"] @ pts_h.T).T[:, :3]
    z = cam_pts[:, 2]
    valid = z > 1e-3
    ui = np.zeros(len(cam_pts), dtype=np.int32)
    vi = np.zeros(len(cam_pts), dtype=np.int32)
    if np.any(valid):
        u = camera["fx"] * cam_pts[valid, 0] / z[valid] + camera["cx"]
        v = camera["fy"] * cam_pts[valid, 1] / z[valid] + camera["cy"]
        ui[valid] = np.round(u).astype(np.int32)
        vi[valid] = np.round(v).astype(np.int32)
    in_bounds = valid & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    return ui[in_bounds], vi[in_bounds], z[in_bounds], int(valid.sum())


def visible_target_points_by_depth(
    all_points_xyz: np.ndarray,
    target_points_xyz: np.ndarray,
    camera: dict,
    image_shape: tuple[int, int],
    depth_tolerance: float,
):
    """保留在像素深度竞争中没有被其他场景点遮挡的目标投影点。"""
    h, w = image_shape
    all_u, all_v, all_z, _ = project_points_with_depth(all_points_xyz, camera, image_shape)
    if len(all_z) == 0:
        return np.zeros((0, 2), dtype=np.int32), 0.0

    all_flat = all_v * w + all_u
    order = np.lexsort((all_z, all_flat))
    flat_sorted = all_flat[order]
    z_sorted = all_z[order]
    unique_flat, first_idx = np.unique(flat_sorted, return_index=True)
    depth_map = np.full(h * w, np.inf, dtype=np.float64)
    depth_map[unique_flat] = z_sorted[first_idx]

    tgt_u, tgt_v, tgt_z, tgt_valid_count = project_points_with_depth(target_points_xyz, camera, image_shape)
    if len(tgt_z) == 0:
        return np.zeros((0, 2), dtype=np.int32), 0.0

    tgt_flat = tgt_v * w + tgt_u
    visible = tgt_z <= depth_map[tgt_flat] + depth_tolerance
    points_2d = np.stack([tgt_u[visible], tgt_v[visible]], axis=1)
    return points_2d, float(len(points_2d)) / max(tgt_valid_count, 1)


def projection_hull_mask(points_2d: np.ndarray, image_shape: tuple[int, int], dilation: int = 7) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points_2d) >= 3:
        hull = cv2.convexHull(points_2d.astype(np.int32).reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        for x, y in points_2d:
            cv2.circle(mask, (int(x), int(y)), 2, 255, -1)
    if dilation > 0 and mask.any():
        mask = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8), iterations=1)
    return mask > 0


def projection_support_mask(
    points_2d: np.ndarray,
    image_shape: tuple[int, int],
    radius: int,
    close_kernel: int,
) -> np.ndarray:
    """用目标可见点生成局部支持区域，避免 SAM 吞入没有 3D 点支持的桌面杂物。"""
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points_2d) == 0:
        return mask.astype(bool)
    radius = max(int(radius), 1)
    for x, y in points_2d.astype(np.int32):
        cv2.circle(mask, (int(x), int(y)), radius, 255, thickness=-1)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask > 0


def bbox_from_points(points_2d: np.ndarray, image_shape: tuple[int, int], padding: int) -> np.ndarray:
    h, w = image_shape
    x0, y0 = points_2d.min(axis=0)
    x1, y1 = points_2d.max(axis=0)
    return np.array(
        [
            max(0, x0 - padding),
            max(0, y0 - padding),
            min(w - 1, x1 + padding),
            min(h - 1, y1 + padding),
        ],
        dtype=np.float32,
    )


def sample_points(points_2d: np.ndarray, count: int) -> np.ndarray:
    if len(points_2d) <= count:
        return points_2d.astype(np.float32)
    center = points_2d.mean(axis=0, keepdims=True)
    selected = [int(np.argmax(np.linalg.norm(points_2d - center, axis=1)))]
    min_dist = np.full(len(points_2d), np.inf, dtype=np.float64)
    for _ in range(1, count):
        dist = np.linalg.norm(points_2d - points_2d[selected[-1]], axis=1)
        min_dist = np.minimum(min_dist, dist)
        selected.append(int(np.argmax(min_dist)))
    return points_2d[selected].astype(np.float32)


def sample_negative_points(box: np.ndarray, image_shape: tuple[int, int], count: int, margin: int = 16) -> np.ndarray:
    h, w = image_shape
    x0, y0, x1, y1 = box.astype(int)
    candidates = [
        (max(0, x0 - margin), y0),
        (min(w - 1, x1 + margin), y0),
        (x0, max(0, y0 - margin)),
        (x1, min(h - 1, y1 + margin)),
        (max(0, x0 - margin), max(0, y0 - margin)),
        (min(w - 1, x1 + margin), max(0, y0 - margin)),
        (max(0, x0 - margin), min(h - 1, y1 + margin)),
        (min(w - 1, x1 + margin), min(h - 1, y1 + margin)),
    ]
    return np.asarray(candidates[:count], dtype=np.float32)


def sample_internal_negative_points(
    box: np.ndarray,
    support_mask: np.ndarray,
    count: int,
    grid: int = 8,
) -> np.ndarray:
    """在投影 box 内采样没有目标点支持的位置，作为 SAM 负点。"""
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    h, w = support_mask.shape
    x0, y0, x1, y1 = box.astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 2), dtype=np.float32)

    xs = np.linspace(x0, x1, max(grid, 2)).astype(np.int32)
    ys = np.linspace(y0, y1, max(grid, 2)).astype(np.int32)
    candidates = []
    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
    for y in ys:
        for x in xs:
            if not support_mask[y, x]:
                candidates.append((x, y))
    if not candidates:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(candidates, dtype=np.float32)
    order = np.argsort(np.linalg.norm(pts - center[None, :], axis=1))[::-1]
    return pts[order[:count]]


def score_view_candidate(points_2d: np.ndarray, image_shape: tuple[int, int], visible_ratio: float) -> tuple[float, dict]:
    h, w = image_shape
    box = bbox_from_points(points_2d, image_shape, padding=0)
    x0, y0, x1, y1 = box
    box_w = max(float(x1 - x0), 1.0)
    box_h = max(float(y1 - y0), 1.0)
    box_area_ratio = (box_w * box_h) / max(float(w * h), 1.0)
    center = np.array([(x0 + x1) * 0.5 / max(w, 1), (y0 + y1) * 0.5 / max(h, 1)])
    center_dist = float(np.linalg.norm(center - np.array([0.5, 0.5])))
    # 这里只做空间粗过滤：能看到足够多 bbox/indices 内的点，就值得送入 SAM。
    # 贴边、居中、面积大小只记录用于诊断，不参与扣分。
    score = visible_ratio * 100000.0 + float(len(points_2d))
    stats = {
        "view_score": float(score),
        "visible_points": int(len(points_2d)),
        "visible_ratio": float(visible_ratio),
        "projection_box": box.tolist(),
        "projection_box_area_ratio": float(box_area_ratio),
        "center_distance": center_dist,
    }
    return float(score), stats


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = a.astype(float)
    bx0, by0, bx1, by1 = b.astype(float)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(area_a + area_b - inter, 1e-6)


def score_mask(
    mask: np.ndarray,
    points_2d: np.ndarray,
    hull_mask: np.ndarray,
    support_mask: np.ndarray,
    box: np.ndarray,
    unsupported_weight: float,
) -> tuple[float, dict]:
    h, w = mask.shape
    px = np.clip(points_2d[:, 0].astype(int), 0, w - 1)
    py = np.clip(points_2d[:, 1].astype(int), 0, h - 1)
    point_cover = float(mask[py, px].mean()) if len(points_2d) else 0.0
    mask_area = float(mask.sum())
    hull_area = float(hull_mask.sum())
    inter = float(np.logical_and(mask, hull_mask).sum())
    hull_iou = inter / max(mask_area + hull_area - inter, 1.0)
    area_ratio = mask_area / max(hull_area, 1.0)
    ratio_penalty = max(0.0, abs(np.log(max(area_ratio, 1e-6))) - np.log(4.0))
    x0, y0, x1, y1 = box.astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    in_box = np.zeros_like(mask, dtype=bool)
    if x1 >= x0 and y1 >= y0:
        in_box[y0 : y1 + 1, x0 : x1 + 1] = True
    outside_ratio = float(np.logical_and(mask, ~in_box).sum()) / max(mask_area, 1.0)
    unsupported_ratio = float(np.logical_and(mask, ~support_mask).sum()) / max(mask_area, 1.0)
    touches_border = mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    score = point_cover * 3.0 + hull_iou * 2.0 - outside_ratio - ratio_penalty - unsupported_ratio * unsupported_weight
    stats = {
        "point_cover": point_cover,
        "hull_iou": hull_iou,
        "area_ratio": area_ratio,
        "outside_ratio": outside_ratio,
        "unsupported_ratio": unsupported_ratio,
        "touches_border": bool(touches_border),
        "mask_area": int(mask_area),
        "score": float(score),
    }
    return float(score), stats


def choose_sam_mask_box_only(
    sam_predictor: SamPredictor,
    image_rgb: np.ndarray,
    box: np.ndarray,
    min_mask_pixels: int,
) -> tuple[np.ndarray | None, dict | None]:
    """仅用 2D AABB 提示 SAM，不使用正负点。"""
    sam_predictor.set_image(image_rgb)
    masks, sam_scores, _ = sam_predictor.predict(
        box=box,
        multimask_output=True,
    )
    best = None
    for idx, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        mask_area = int(mask_bool.sum())
        if mask_area < min_mask_pixels:
            continue
        score = float(sam_scores[idx])
        stats = {
            "sam_score": score,
            "mask_area": mask_area,
            "box": box.tolist(),
            "accepted": True,
        }
        if best is None or score > best[0]:
            best = (score, mask_bool, stats)
    if best is None:
        return None, None
    return best[1], best[2]


def project_bbox_aabb_box(
    bbox_corners: np.ndarray,
    camera: dict,
    image_shape: tuple[int, int],
    padding: int,
) -> tuple[np.ndarray | None, np.ndarray, float]:
    """将 3D bbox 8 角点投影到图像，取可见点的 2D AABB。"""
    corners = np.asarray(bbox_corners, dtype=np.float64).reshape(-1, 3)
    points_2d, visible_ratio = project_points(corners, camera, image_shape)
    if len(points_2d) < 2:
        return None, points_2d, visible_ratio
    box = bbox_from_points(points_2d, image_shape, padding)
    x0, y0, x1, y1 = box.astype(int)
    if x1 <= x0 or y1 <= y0:
        return None, points_2d, visible_ratio
    return box, points_2d, visible_ratio


def choose_sam_mask(
    sam_predictor: SamPredictor,
    image_rgb: np.ndarray,
    box: np.ndarray,
    pos_points: np.ndarray,
    neg_points: np.ndarray,
    points_2d: np.ndarray,
    hull_mask: np.ndarray,
    support_mask: np.ndarray,
    unsupported_weight: float,
):
    sam_predictor.set_image(image_rgb)
    if len(neg_points):
        point_coords = np.concatenate([pos_points, neg_points], axis=0)
        point_labels = np.concatenate(
            [np.ones(len(pos_points), dtype=np.int32), np.zeros(len(neg_points), dtype=np.int32)],
            axis=0,
        )
    else:
        point_coords = pos_points
        point_labels = np.ones(len(pos_points), dtype=np.int32)
    masks, sam_scores, _ = sam_predictor.predict(
        point_coords=point_coords if len(point_coords) else None,
        point_labels=point_labels if len(point_coords) else None,
        box=box,
        multimask_output=True,
    )
    best = None
    for idx, mask in enumerate(masks):
        score, stats = score_mask(mask.astype(bool), points_2d, hull_mask, support_mask, box, unsupported_weight)
        score += float(sam_scores[idx]) * 0.25
        stats["sam_score"] = float(sam_scores[idx])
        if best is None or score > best[0]:
            best = (score, mask.astype(bool), stats)
    return best[1], best[2]


def maybe_grounding_boxes(model, image_rgb: np.ndarray, class_name: str, box: np.ndarray, args) -> list[np.ndarray]:
    if model is None:
        return []
    detections = model.predict_with_classes(
        image=image_rgb,
        classes=[class_name],
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    boxes = []
    for det_box in getattr(detections, "xyxy", []):
        det_box = det_box.astype(np.float32)
        if box_iou(det_box, box) >= args.dino_min_iou:
            boxes.append(det_box)
    return boxes


def transfer_file(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(description="single model Grounded-SAM masks from 3D bbox/features")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--single-model-json", type=Path, default=None)
    parser.add_argument("--source-ply", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-mode", choices=("bbox", "indices", "auto"), default="bbox")
    parser.add_argument(
        "--mode",
        choices=("sam_points", "grounded_sam", "hybrid", "bbox"),
        default="hybrid",
        help="mask 生成模式；bbox=3D bbox 投影 2D AABB 后仅 box 提示 SAM 分割",
    )
    parser.add_argument(
        "--sam-mode",
        dest="mode",
        choices=("sam_points", "grounded_sam", "hybrid", "bbox"),
        help="同 --mode，兼容 shell/JSON 传参",
    )
    parser.add_argument("--sam_checkpoint_path", default="weights/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--groundingdino_checkpoint_path", default="weights/groundingdino_swint_ogc.pth", type=str)
    parser.add_argument("--groundingdino_config_path", default="weights/GroundingDINO_SwinT_OGC.py", type=str)
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.35)
    parser.add_argument("--dino-min-iou", type=float, default=0.15)
    parser.add_argument("--box-padding", type=int, default=16)
    parser.add_argument("--positive-points", type=int, default=24)
    parser.add_argument("--negative-points", type=int, default=8)
    parser.add_argument("--internal-negative-points", type=int, default=16, help="在投影 box 内无 3D 支持区域采样的 SAM 负点数")
    parser.add_argument("--min-visible-points", type=int, default=20)
    parser.add_argument("--prefilter-views", type=int, default=0, help="空间粗过滤后最多送入 SAM 的视角数，0 表示所有可见视角")
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--projection-dilation", type=int, default=7)
    parser.add_argument("--support-dilation", type=int, default=9, help="目标可见投影点生成支持区域的膨胀半径")
    parser.add_argument("--support-close-kernel", type=int, default=15, help="目标支持区域闭运算核大小")
    parser.add_argument("--depth-tolerance", type=float, default=0.03, help="目标点与全场景最近深度竞争的容差")
    parser.add_argument("--no-depth-occlusion-check", action="store_true", help="关闭全场景深度遮挡检查")
    parser.add_argument("--constrain-to-3d-support", action="store_true", help="把最终 SAM mask 裁剪到目标 3D 支持区域（默认关闭，避免损伤 ShapeR 几何）")
    parser.add_argument(
        "--no-constrain-to-3d-support",
        action="store_false",
        dest="constrain_to_3d_support",
        help="兼容旧参数：保持最终 SAM mask 完整，不做 3D 支持区域裁剪",
    )
    parser.add_argument("--unsupported-weight", type=float, default=2.0, help="SAM mask 落在无 3D 支持区域的评分惩罚权重")
    parser.add_argument("--min-point-cover", type=float, default=0.55)
    parser.add_argument("--max-area-ratio", type=float, default=8.0)
    parser.add_argument("--start-view", type=int, default=0, help="调试用：从排序后的第几个视角开始处理")
    parser.add_argument("--max-views", type=int, default=0, help="调试用：最多处理多少张图，0 表示全量")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--annotated-images-path", type=Path, default=None)
    parser.add_argument("--progress-path", type=Path, default=None)
    parser.set_defaults(constrain_to_3d_support=False)
    return parser.parse_args()


def run_bbox_mask_mode(
    args,
    paths: dict[str, Path],
    single_model_json: Path,
    source_ply: Path,
    config: dict,
    class_name: str,
    class_idx: int,
) -> None:
    """3D bbox 投影为 2D AABB，仅用 box 提示 SAM 生成 mask。"""
    if args.selection_mode != "bbox":
        print(f"[bbox mode] 强制使用 selection-mode=bbox（忽略 {args.selection_mode}）")
    selection_mode = "bbox"
    bbox_corners = np.asarray(config["bbox"], dtype=np.float64)

    output_dir = args.output_dir.resolve()
    images_out = output_dir / "images"
    mask_png_out = output_dir / "mask"
    saga_masks_out = output_dir / "sam_masks"
    labels_out = output_dir / "labels"
    for path in (images_out, mask_png_out, saga_masks_out, labels_out):
        path.mkdir(parents=True, exist_ok=True)
    if args.annotated_images_path:
        args.annotated_images_path.mkdir(parents=True, exist_ok=True)

    torch.save(words_to_tensors(args.classes), labels_out / "label_features.pt")

    vertices, vertex_props = read_binary_vertex_ply(source_ply)
    point_mask, selected_by = select_points_from_json(vertices, config, selection_mode)
    if int(point_mask.sum()) == 0:
        raise RuntimeError(f"{single_model_json} 的 bbox 没有在 {source_ply} 中选中任何点")

    ply_out = output_dir / SINGLE_MODEL_FILTED_PLY_NAME
    write_binary_vertex_ply(ply_out, vertices[point_mask], vertex_props)

    cameras = load_colmap_cameras(paths["colmap_dir"])
    image_items = sorted(cameras.items())
    if args.start_view > 0:
        image_items = image_items[args.start_view :]
    if args.max_views > 0:
        image_items = image_items[: args.max_views]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境没有可用 GPU")

    print(f"Initializing SAM on {device} for bbox-AABB mode...")
    sam_checkpoint_path = resolve_weight_path(args.sam_checkpoint_path)
    sam = sam_model_registry[args.sam_arch](checkpoint=sam_checkpoint_path).to(device)
    sam_predictor = SamPredictor(sam)

    selected_views = []
    skipped = {"no_image": 0, "low_visibility": 0, "bad_mask": 0}
    total_views = len(image_items)

    print(f"bbox→AABB→SAM：处理 {total_views} 个视角...")
    for index, (stem, camera) in enumerate(tqdm(image_items)):
        if args.progress_path:
            args.progress_path.parent.mkdir(parents=True, exist_ok=True)
            args.progress_path.write_text(str((index + 1) * 100 // max(total_views, 1)))

        image_path = resolve_image_path(paths["image_dir"], camera["name"])
        if image_path is None:
            skipped["no_image"] += 1
            continue
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            skipped["no_image"] += 1
            continue

        h, w = image_bgr.shape[:2]
        projection_box, corner_points_2d, visible_ratio = project_bbox_aabb_box(
            bbox_corners,
            camera,
            (h, w),
            args.box_padding,
        )
        if projection_box is None or len(corner_points_2d) < 2:
            skipped["low_visibility"] += 1
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        final_mask, stats = choose_sam_mask_box_only(
            sam_predictor,
            image_rgb,
            projection_box,
            args.min_mask_pixels,
        )
        if final_mask is None or stats is None:
            skipped["bad_mask"] += 1
            continue

        stats["visible_corners"] = int(len(corner_points_2d))
        stats["visible_ratio"] = float(visible_ratio)

        image_dst = images_out / image_path.name
        mask_dst = mask_png_out / f"{stem}.png"
        transfer_file(image_path, image_dst, args.symlink)
        Image.fromarray(final_mask.astype(np.uint8) * 255, mode="L").save(mask_dst)
        torch.save(torch.from_numpy(final_mask[None, ...]).bool(), saga_masks_out / f"{stem}.pt")
        torch.save(torch.tensor([class_idx], dtype=torch.long), labels_out / f"{stem}.pt")

        if args.annotated_images_path:
            annotated = image_bgr.copy()
            x0, y0, x1, y1 = projection_box.astype(int)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
            overlay = annotated.copy()
            overlay[final_mask] = (0, 255, 0)
            annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
            cv2.imwrite(str(args.annotated_images_path / f"{stem}.jpg"), annotated)

        selected_views.append(
            {
                "image_name": image_path.name,
                "mask_name": mask_dst.name,
                "visible_projected_pixels": int(len(corner_points_2d)),
                "visible_ratio": float(visible_ratio),
                "mask_pixels": int(final_mask.sum()),
                "bbox_aabb_stats": stats,
            }
        )

    if not selected_views:
        raise RuntimeError(f"bbox→SAM 模式没有导出任何有效视角，skipped={skipped}")

    summary = {
        "scene_dir": str(args.scene_dir.resolve()),
        "single_model_json": str(single_model_json.resolve()),
        "source_ply": str(source_ply.resolve()),
        "class_name": class_name,
        "class_index": class_idx,
        "caption": caption_from_config(config),
        "selection_mode": selection_mode,
        "selected_by": selected_by,
        "mask_generator": "bbox_aabb_sam_single_model",
        "mode": args.mode,
        "point_count": int(point_mask.sum()),
        "view_count": len(selected_views),
        "visible_view_count": len(selected_views),
        "prefilter_view_count": total_views,
        "output_dir": str(output_dir),
        "ply_path": str(ply_out),
        "image_dir": str(images_out),
        "mask_dir": str(mask_png_out),
        "saga_masks_dir": str(saga_masks_out),
        "labels_dir": str(labels_out),
        "colmap_dir": str(paths["colmap_dir"].parent),
        "single_model_info": config,
        "views": selected_views,
        "skipped": skipped,
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[single_model] class={class_name}, points={int(point_mask.sum())}, views={len(selected_views)}")
    print(f"点云: {ply_out}")
    print(f"图像: {images_out}")
    print(f"Mask: {mask_png_out}")
    print(f"meta: {output_dir / 'meta.json'}")


def main():
    args = parse_args()
    paths = resolve_scene_paths(args.scene_dir)
    single_model_json = args.single_model_json or paths["single_model_json"]
    source_ply = args.source_ply or paths["source_ply"]
    if not single_model_json.exists():
        raise FileNotFoundError(f"缺少 single model json: {single_model_json}")
    if not source_ply.exists():
        raise FileNotFoundError(f"缺少输入点云: {source_ply}")

    config = load_single_model_config(single_model_json)
    class_name = config["class"]
    class_idx = args.classes.index(class_name) if class_name in args.classes else 0
    if class_name not in args.classes:
        args.classes = [class_name] + list(args.classes)
        class_idx = 0

    if args.mode == "bbox":
        run_bbox_mask_mode(
            args=args,
            paths=paths,
            single_model_json=single_model_json,
            source_ply=source_ply,
            config=config,
            class_name=class_name,
            class_idx=class_idx,
        )
        return

    output_dir = args.output_dir.resolve()
    images_out = output_dir / "images"
    mask_png_out = output_dir / "mask"
    saga_masks_out = output_dir / "sam_masks"
    labels_out = output_dir / "labels"
    for path in (images_out, mask_png_out, saga_masks_out, labels_out):
        path.mkdir(parents=True, exist_ok=True)
    if args.annotated_images_path:
        args.annotated_images_path.mkdir(parents=True, exist_ok=True)

    torch.save(words_to_tensors(args.classes), labels_out / "label_features.pt")

    vertices, vertex_props = read_binary_vertex_ply(source_ply)
    point_mask, selected_by = select_points_from_json(vertices, config, args.selection_mode)
    if int(point_mask.sum()) == 0:
        raise RuntimeError(f"{single_model_json} 没有在 {source_ply} 中选中任何点")
    points_xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    instance_points = points_xyz[point_mask]
    ply_out = output_dir / SINGLE_MODEL_FILTED_PLY_NAME
    write_binary_vertex_ply(ply_out, vertices[point_mask], vertex_props)

    cameras = load_colmap_cameras(paths["colmap_dir"])
    image_items = sorted(cameras.items())
    if args.start_view > 0:
        image_items = image_items[args.start_view :]
    if args.max_views > 0:
        image_items = image_items[: args.max_views]

    selected_views = []
    skipped = {"no_image": 0, "low_visibility": 0, "bad_mask": 0, "prefiltered": 0}
    view_candidates = []

    print("空间方位预筛选视角...")
    for stem, camera in tqdm(image_items):
        image_path = resolve_image_path(paths["image_dir"], camera["name"])
        if image_path is None:
            skipped["no_image"] += 1
            continue
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            skipped["no_image"] += 1
            continue
        h, w = image_bgr.shape[:2]
        if args.no_depth_occlusion_check:
            points_2d, visible_ratio = project_points(instance_points, camera, (h, w))
        else:
            points_2d, visible_ratio = visible_target_points_by_depth(
                points_xyz,
                instance_points,
                camera,
                (h, w),
                args.depth_tolerance,
            )
        if len(points_2d) < args.min_visible_points:
            skipped["low_visibility"] += 1
            continue
        view_score, view_stats = score_view_candidate(
            points_2d,
            (h, w),
            visible_ratio,
        )
        view_candidates.append(
            {
                "stem": stem,
                "camera": camera,
                "image_path": image_path,
                "image_shape": (h, w),
                "points_2d": points_2d,
                "visible_ratio": visible_ratio,
                "view_score": view_score,
                "view_stats": view_stats,
            }
        )

    view_candidates.sort(key=lambda item: item["view_score"], reverse=True)
    visible_view_count = len(view_candidates)
    if args.prefilter_views > 0 and len(view_candidates) > args.prefilter_views:
        skipped["prefiltered"] = len(view_candidates) - args.prefilter_views
        view_candidates = view_candidates[: args.prefilter_views]
    print(f"空间预筛选: 可见视角 {visible_view_count}，送入 SAM {len(view_candidates)}")

    if not view_candidates:
        raise RuntimeError(f"空间预筛选后没有候选视角，skipped={skipped}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境没有可用 GPU")

    print(f"Initializing SAM on {device}...")
    sam_checkpoint_path = resolve_weight_path(args.sam_checkpoint_path)
    sam = sam_model_registry[args.sam_arch](checkpoint=sam_checkpoint_path).to(device)
    sam_predictor = SamPredictor(sam)

    grounding_model = None
    if args.mode in ("grounded_sam", "hybrid"):
        if Model is None:
            print("GroundingDINO 不可用，回退到 sam_points")
        else:
            print("Initializing GroundingDINO...")
            grounding_model = Model(
                model_config_path=resolve_weight_path(args.groundingdino_config_path),
                model_checkpoint_path=resolve_weight_path(args.groundingdino_checkpoint_path),
            )

    for index, candidate in enumerate(tqdm(view_candidates)):
        if args.progress_path:
            args.progress_path.parent.mkdir(parents=True, exist_ok=True)
            args.progress_path.write_text(str((index + 1) * 100 // max(len(view_candidates), 1)))

        stem = candidate["stem"]
        image_path = candidate["image_path"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            skipped["no_image"] += 1
            continue

        h, w = candidate["image_shape"]
        points_2d = candidate["points_2d"]
        visible_ratio = candidate["visible_ratio"]
        hull_mask = projection_hull_mask(points_2d, (h, w), args.projection_dilation)
        support_mask = projection_support_mask(
            points_2d,
            (h, w),
            args.support_dilation,
            args.support_close_kernel,
        )
        projection_box = bbox_from_points(points_2d, (h, w), args.box_padding)
        pos_points = sample_points(points_2d, args.positive_points)
        outer_neg_points = sample_negative_points(projection_box, (h, w), args.negative_points)
        inner_neg_points = sample_internal_negative_points(
            projection_box,
            support_mask,
            args.internal_negative_points,
        )
        neg_points = np.concatenate([outer_neg_points, inner_neg_points], axis=0)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        candidate_boxes = []
        if args.mode in ("grounded_sam", "hybrid"):
            candidate_boxes.extend(maybe_grounding_boxes(grounding_model, image_rgb, class_name, projection_box, args))
        if args.mode in ("sam_points", "hybrid") or not candidate_boxes:
            candidate_boxes.append(projection_box)

        best = None
        for candidate_box in candidate_boxes:
            mask, stats = choose_sam_mask(
                sam_predictor,
                image_rgb,
                candidate_box,
                pos_points,
                neg_points,
                points_2d,
                hull_mask,
                support_mask,
                args.unsupported_weight,
            )
            stats["box"] = candidate_box.tolist()
            stats["visible_points"] = int(len(points_2d))
            stats["visible_ratio"] = float(visible_ratio)
            stats["view_score"] = float(candidate["view_score"])
            stats["support_pixels"] = int(support_mask.sum())
            stats["negative_points"] = int(len(neg_points))
            accept = (
                stats["mask_area"] >= args.min_mask_pixels
                and stats["point_cover"] >= args.min_point_cover
                and stats["area_ratio"] <= args.max_area_ratio
            )
            stats["accepted"] = bool(accept)
            if best is None or stats["score"] > best[0]:
                best = (stats["score"], mask, stats)

        if best is None or not best[2]["accepted"]:
            skipped["bad_mask"] += 1
            continue

        final_mask = best[1]
        stats = best[2]
        if args.constrain_to_3d_support:
            final_mask = np.logical_and(final_mask, support_mask)
            stats["constrained_to_3d_support"] = True
            stats["constrained_mask_area"] = int(final_mask.sum())
            if stats["constrained_mask_area"] < args.min_mask_pixels:
                skipped["bad_mask"] += 1
                continue
        else:
            stats["constrained_to_3d_support"] = False
        image_dst = images_out / image_path.name
        mask_dst = mask_png_out / f"{stem}.png"
        transfer_file(image_path, image_dst, args.symlink)
        Image.fromarray(final_mask.astype(np.uint8) * 255, mode="L").save(mask_dst)
        torch.save(torch.from_numpy(final_mask[None, ...]).bool(), saga_masks_out / f"{stem}.pt")
        torch.save(torch.tensor([class_idx], dtype=torch.long), labels_out / f"{stem}.pt")

        if args.annotated_images_path:
            annotated = image_bgr.copy()
            x0, y0, x1, y1 = np.asarray(stats["box"], dtype=int)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
            overlay = annotated.copy()
            overlay[final_mask] = (0, 255, 0)
            annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
            cv2.imwrite(str(args.annotated_images_path / f"{stem}.jpg"), annotated)

        selected_views.append(
            {
                "image_name": image_path.name,
                "mask_name": mask_dst.name,
                "visible_projected_pixels": int(len(points_2d)),
                "visible_ratio": float(visible_ratio),
                "mask_pixels": int(final_mask.sum()),
                "view_prefilter": candidate["view_stats"],
                "sam_stats": stats,
            }
        )

    if not selected_views:
        raise RuntimeError(f"没有导出任何有效视角，skipped={skipped}")

    summary = {
        "scene_dir": str(args.scene_dir.resolve()),
        "single_model_json": str(single_model_json.resolve()),
        "source_ply": str(source_ply.resolve()),
        "class_name": class_name,
        "class_index": class_idx,
        "caption": caption_from_config(config),
        "selection_mode": args.selection_mode,
        "selected_by": selected_by,
        "mask_generator": "grounded_sam_single_model",
        "mode": args.mode,
        "point_count": int(point_mask.sum()),
        "view_count": len(selected_views),
        "visible_view_count": visible_view_count,
        "prefilter_view_count": len(view_candidates),
        "output_dir": str(output_dir),
        "ply_path": str(ply_out),
        "image_dir": str(images_out),
        "mask_dir": str(mask_png_out),
        "saga_masks_dir": str(saga_masks_out),
        "labels_dir": str(labels_out),
        "colmap_dir": str(paths["colmap_dir"].parent),
        "single_model_info": config,
        "views": selected_views,
        "skipped": skipped,
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[single_model] class={class_name}, points={int(point_mask.sum())}, views={len(selected_views)}")
    print(f"点云: {ply_out}")
    print(f"图像: {images_out}")
    print(f"Mask: {mask_png_out}")
    print(f"SAGA masks: {saga_masks_out}")
    print(f"meta: {output_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
