from __future__ import annotations

"""Native V9 mask-to-Gaussian attribution and artifact contract.

The worker-facing primitives here intentionally have no dependency on any
V3--V8 experiment module.  M1 and AM are expressed in the same per-pixel
normalised mass units, missing detections are abstentions, and every serialized
fragment carries explicit same-physical-view conflict evidence.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


V9_LIFTING_SCHEMA = "saga-v9-native-lifting-bank-v1"
V9_LIFTING_IDENTITY_SCHEMA = "saga-v9-native-lifting-identity-v1"
DEFAULT_CLASSES = (
    "chair", "table", "plant", "flower", "foliage", "tv", "painting",
    "sofa", "cabinet", "bed", "wall", "floor", "ceiling", "person",
    "socket", "remote", "key", "book", "lighting", "switch", "door",
    "window", "lamp", "speaker", "computer", "fan", "refrigerator",
    "robot", "cup", "vase", "phone", "trash can",
)


@dataclass(frozen=True)
class FragmentConfig:
    full_min_inside_mass: float = 0.5
    core_min_inside_mass: float = 2.0
    core_min_inside_ratio: float = 0.50
    fragment_min_core: int = 3
    fragment_min_full: int = 10


def _source_file_identity(path: Path) -> dict[str, Any]:
    target = Path(path).resolve()
    stat = target.stat()
    return {
        "path": str(target),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_lifting_identity(
    *,
    scene_id: str,
    git_commit: str,
    feature_ply: Path,
    feature_record: Path,
    label_features: Path,
    segment_everything_root: Path,
    classes: Sequence[str],
    config: FragmentConfig = FragmentConfig(),
) -> dict[str, Any]:
    """Build the no-hash source identity used for skip and cleanup decisions."""

    commit = str(git_commit).strip()
    if not commit:
        raise ValueError("lifting git_commit must be non-empty")
    record = json.loads(Path(feature_record).read_text("utf-8"))
    record_identity = record.get("identity")
    if (
        record.get("status") != "complete"
        or record.get("git_commit") != commit
        or not isinstance(record_identity, Mapping)
    ):
        raise ValueError("lifting requires a complete current-commit feature record")
    declared_outputs = record_identity.get("outputs")
    if (
        not isinstance(declared_outputs, Mapping)
        or Path(str(declared_outputs.get("feature_ply", ""))).resolve()
        != Path(feature_ply).resolve()
    ):
        raise ValueError("feature record does not declare the lifting feature PLY")
    summary_path = Path(segment_everything_root).resolve() / "summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    if summary.get("schema") != "saga-v9-segment-everything-v1":
        raise ValueError("lifting requires a registered SAM-everything summary")
    sam_identity = {
        key: summary.get(key)
        for key in (
            "schema",
            "image_root",
            "output_root",
            "sam_arch",
            "config",
            "image_count",
            "mask_count",
            "images",
        )
    }
    return {
        "schema": V9_LIFTING_IDENTITY_SCHEMA,
        "scene_id": str(scene_id),
        "git_commit": commit,
        "feature_ply": _source_file_identity(Path(feature_ply)),
        "feature_record_path": str(Path(feature_record).resolve()),
        "feature_record_identity": dict(record_identity),
        "label_features": _source_file_identity(Path(label_features)),
        "segment_everything_root": str(Path(segment_everything_root).resolve()),
        "segment_everything": sam_identity,
        "classes": list(map(str, classes)),
        "fragment_config": {
            key: value for key, value in vars(config).items()
        },
    }


@dataclass(frozen=True)
class AttributionMass:
    source: str
    inside_mass: np.ndarray
    visible_mass: np.ndarray
    valid_pixel_count: int
    abstained: bool = False

    @property
    def mask_count(self) -> int:
        return int(self.inside_mass.shape[0])


@dataclass(frozen=True)
class Fragment:
    fragment_id: int
    frame_id: int
    mask_index: int
    full_ids: np.ndarray
    core_ids: np.ndarray
    full_mass: np.ndarray
    core_mass: np.ndarray
    core_ratio: np.ndarray
    conflict_ratio: float


@dataclass(frozen=True)
class ChannelBatch:
    mask_indices: tuple[int, ...]
    targets: np.ndarray


@dataclass(frozen=True)
class ObjectiveTargets:
    mask_indices: tuple[int, ...]
    inside_coefficients: np.ndarray
    visible_coefficient: np.ndarray
    valid_pixels: np.ndarray


def _validated_mass(
    source: str,
    inside: np.ndarray,
    visible: np.ndarray,
    valid_pixel_count: int,
    *,
    abstained: bool,
) -> AttributionMass:
    inside = np.asarray(inside, dtype=np.float64)
    visible = np.asarray(visible, dtype=np.float64)
    if inside.ndim != 2 or visible.ndim != 1 or inside.shape[1] != len(visible):
        raise ValueError("inside mass must be MxN and visible mass must be N")
    if np.any(~np.isfinite(inside)) or np.any(~np.isfinite(visible)):
        raise ValueError("attribution mass must be finite")
    if np.any(inside < -1e-7) or np.any(visible < -1e-7):
        raise ValueError("attribution mass must be non-negative")
    inside = np.maximum(inside, 0.0)
    visible = np.maximum(visible, 0.0)
    tolerance = 5e-5 * np.maximum(visible[None, :], 1.0)
    if inside.size and np.any(inside - visible[None, :] > tolerance):
        raise ValueError("inside mass cannot exceed visible mass")
    return AttributionMass(
        str(source),
        np.minimum(inside, visible[None, :]),
        visible,
        int(valid_pixel_count),
        bool(abstained),
    )


def mass_from_max_contributor(
    max_id: np.ndarray,
    max_weight: np.ndarray,
    masks: np.ndarray | None,
    point_count: int,
) -> AttributionMass:
    """Reduce corrected ``alpha*T_prev`` winners; empty pixels contribute 0."""

    ids = np.asarray(max_id)
    weights = np.asarray(max_weight, dtype=np.float64)
    if ids.ndim != 2 or weights.shape != ids.shape or point_count <= 0:
        raise ValueError("M1 needs matching HxW ID/weight images and positive N")
    abstained = masks is None
    mask_array = (
        np.zeros((0, *ids.shape), dtype=bool)
        if masks is None
        else np.asarray(masks, dtype=bool)
    )
    if mask_array.ndim != 3 or mask_array.shape[1:] != ids.shape:
        raise ValueError("masks must be MxHxW and match M1 images")
    flat_id = ids.reshape(-1).astype(np.int64, copy=False)
    flat_weight = weights.reshape(-1)
    valid = (
        (flat_id >= 0)
        & (flat_id < int(point_count))
        & np.isfinite(flat_weight)
        & (flat_weight > 0)
    )
    # M1 is a one-hot normalized attribution: weight selects the winner and
    # rejects empty pixels, while each valid pixel contributes unit mass.
    visible = np.bincount(flat_id[valid], minlength=point_count).astype(np.float64)
    inside = np.zeros((len(mask_array), point_count), dtype=np.float64)
    for mask_index, mask in enumerate(mask_array):
        selected = valid & mask.reshape(-1)
        inside[mask_index] = np.bincount(
            flat_id[selected], minlength=point_count
        )
    return _validated_mass(
        "M1", inside, visible, int(np.count_nonzero(valid)), abstained=abstained
    )


def iter_mask_batches(masks: np.ndarray) -> Iterator[ChannelBatch]:
    values = np.asarray(masks, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("masks must be MxHxW")
    for start in range(0, len(values), 3):
        stop = min(start + 3, len(values))
        targets = np.zeros((3, *values.shape[1:]), dtype=np.float32)
        targets[: stop - start] = values[start:stop]
        yield ChannelBatch(tuple(range(start, stop)), targets)


def build_objectives(batch: ChannelBatch, opacity: np.ndarray) -> ObjectiveTargets:
    opacity = np.asarray(opacity, dtype=np.float64)
    if batch.targets.shape != (3, *opacity.shape):
        raise ValueError("opacity and mask batch shapes differ")
    valid = np.isfinite(opacity) & (opacity > 1e-8)
    inverse = np.zeros_like(opacity)
    inverse[valid] = 1.0 / opacity[valid]
    inside = np.asarray(batch.targets, dtype=np.float64) * inverse[None]
    inside[:, ~valid] = 0.0
    return ObjectiveTargets(batch.mask_indices, inside, inverse, valid)


def mass_from_gradients(
    visible_gradient: np.ndarray,
    inside_batches: Sequence[tuple[ChannelBatch, np.ndarray]],
    mask_count: int,
    valid_pixel_count: int,
    *,
    abstained: bool,
) -> AttributionMass:
    visible_raw = np.asarray(visible_gradient, dtype=np.float64)
    visible = visible_raw[:, 0] if visible_raw.ndim == 2 else visible_raw
    if visible.ndim != 1 or np.any(~np.isfinite(visible)) or np.any(visible < -1e-7):
        raise ValueError("visible gradient must be finite non-negative N or NxC")
    inside = np.zeros((int(mask_count), len(visible)), dtype=np.float64)
    seen = np.zeros(int(mask_count), dtype=bool)
    for batch, raw in inside_batches:
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (len(visible), 3) or np.any(~np.isfinite(values)):
            raise ValueError("inside gradients must be finite Nx3")
        if np.any(values < -1e-7):
            raise ValueError("inside gradient mass must be non-negative")
        for channel, mask_index in enumerate(batch.mask_indices):
            if not 0 <= mask_index < mask_count or seen[mask_index]:
                raise ValueError("invalid or repeated AM mask index")
            inside[mask_index] = np.maximum(values[:, channel], 0.0)
            seen[mask_index] = True
    if mask_count and not np.all(seen):
        raise ValueError("missing AM mask gradient")
    return _validated_mass(
        "AM",
        inside,
        np.maximum(visible, 0.0),
        valid_pixel_count,
        abstained=abstained,
    )


def fragments_from_mass(
    mass: AttributionMass,
    frame_id: int,
    stable_mask_offset: int,
    *,
    config: FragmentConfig = FragmentConfig(),
) -> tuple[Fragment, ...]:
    if mass.abstained:
        return ()
    provisional: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    ratios = np.divide(
        mass.inside_mass,
        mass.visible_mass[None],
        out=np.zeros_like(mass.inside_mass),
        where=mass.visible_mass[None] > 0,
    )
    for mask_index, inside in enumerate(mass.inside_mass):
        full = np.flatnonzero(inside >= config.full_min_inside_mass).astype(np.int32)
        core = np.flatnonzero(
            (inside >= config.core_min_inside_mass)
            & (ratios[mask_index] >= config.core_min_inside_ratio)
        ).astype(np.int32)
        if len(full) >= config.fragment_min_full and len(core) >= config.fragment_min_core:
            provisional.append((mask_index, full, core, ratios[mask_index]))
    output: list[Fragment] = []
    for mask_index, full, core, ratio in provisional:
        competitors = [other_core for other_mask, _, other_core, _ in provisional if other_mask != mask_index]
        conflict = (
            np.unique(np.concatenate(competitors))
            if competitors
            else np.empty(0, dtype=np.int32)
        )
        conflict_ratio = float(np.mean(np.isin(core, conflict))) if len(core) else 0.0
        output.append(
            Fragment(
                int(stable_mask_offset + mask_index),
                int(frame_id),
                int(mask_index),
                full,
                core,
                mass.inside_mass[mask_index, full].astype(np.float32),
                mass.inside_mass[mask_index, core].astype(np.float32),
                ratio[core].astype(np.float32),
                conflict_ratio,
            )
        )
    return tuple(output)


def hybrid_fragments(
    maximum: AttributionMass,
    alpha_mass: AttributionMass,
    frame_id: int,
    stable_mask_offset: int,
    *,
    config: FragmentConfig = FragmentConfig(),
) -> tuple[Fragment, ...]:
    """Use M1 for precision core and AM for complete full/visibility support."""

    if maximum.abstained != alpha_mass.abstained:
        raise ValueError("M1 and AM abstention states differ")
    if maximum.inside_mass.shape != alpha_mass.inside_mass.shape:
        raise ValueError("M1 and AM must describe the same masks/Gaussians")
    if maximum.abstained:
        return ()
    m1_ratio = np.divide(
        maximum.inside_mass,
        maximum.visible_mass[None],
        out=np.zeros_like(maximum.inside_mass),
        where=maximum.visible_mass[None] > 0,
    )
    provisional: list[tuple[int, np.ndarray, np.ndarray]] = []
    for mask_index in range(maximum.mask_count):
        full = np.flatnonzero(
            alpha_mass.inside_mass[mask_index] >= config.full_min_inside_mass
        ).astype(np.int32)
        core = np.flatnonzero(
            (maximum.inside_mass[mask_index] >= config.core_min_inside_mass)
            & (m1_ratio[mask_index] >= config.core_min_inside_ratio)
        ).astype(np.int32)
        full = np.union1d(full, core).astype(np.int32)
        if len(full) >= config.fragment_min_full and len(core) >= config.fragment_min_core:
            provisional.append((mask_index, full, core))
    output: list[Fragment] = []
    for mask_index, full, core in provisional:
        competitors = [other for other_mask, _, other in provisional if other_mask != mask_index]
        conflict = np.unique(np.concatenate(competitors)) if competitors else np.empty(0, dtype=np.int32)
        output.append(
            Fragment(
                int(stable_mask_offset + mask_index),
                int(frame_id),
                int(mask_index),
                full,
                core,
                alpha_mass.inside_mass[mask_index, full].astype(np.float32),
                maximum.inside_mass[mask_index, core].astype(np.float32),
                m1_ratio[mask_index, core].astype(np.float32),
                float(np.mean(np.isin(core, conflict))) if len(core) else 0.0,
            )
        )
    return tuple(output)


def pack_ragged(rows: Sequence[np.ndarray], dtype: Any) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    indptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    values = (
        np.concatenate([np.asarray(row, dtype=dtype) for row in rows])
        if int(indptr[-1])
        else np.empty(0, dtype=dtype)
    )
    return indptr, values


def _valid_ragged(indptr: np.ndarray, values: np.ndarray, rows: int, points: int) -> bool:
    return bool(
        np.issubdtype(indptr.dtype, np.integer)
        and np.issubdtype(values.dtype, np.integer)
        and indptr.shape == (rows + 1,)
        and int(indptr[0]) == 0
        and np.all(np.diff(indptr) >= 0)
        and int(indptr[-1]) == len(values)
        and np.all(values >= 0)
        and np.all(values < points)
    )


def lifting_bank_is_complete(
    directory: Path,
    *,
    expected_scene_id: str | None = None,
    expected_git_commit: str | None = None,
    expected_identity: Mapping[str, Any] | None = None,
    expected_feature_record_identity: Mapping[str, Any] | None = None,
) -> bool:
    """Strict native-schema validation, including ``core subset full``."""

    try:
        metadata = json.loads((directory / "lifting_bank.json").read_text("utf-8"))
        if metadata.get("schema") != V9_LIFTING_SCHEMA:
            return False
        if (
            metadata.get("lifting_source") != "M1-core+AM-full"
            or metadata.get("mask_source") != "SAM-everything"
            or metadata.get("feature_source") != "v9-10k-objectbank"
            or metadata.get("config")
            != {key: value for key, value in vars(FragmentConfig()).items()}
        ):
            return False
        if expected_scene_id is not None and metadata.get("scene_id") != expected_scene_id:
            return False
        identity = metadata.get("identity")
        if (
            not isinstance(identity, Mapping)
            or identity.get("schema") != V9_LIFTING_IDENTITY_SCHEMA
            or identity.get("scene_id") != metadata.get("scene_id")
            or identity.get("git_commit") != metadata.get("git_commit")
        ):
            return False
        if expected_git_commit is not None and identity.get("git_commit") != expected_git_commit:
            return False
        if expected_identity is not None and dict(identity) != dict(expected_identity):
            return False
        if (
            expected_feature_record_identity is not None
            and identity.get("feature_record_identity")
            != dict(expected_feature_record_identity)
        ):
            return False
        n = int(metadata["point_count"])
        f = int(metadata["fragment_count"])
        frames = int(metadata["frame_count"])
        semantic_fragments = int(metadata["semantic_fragment_count"])
        classes = metadata.get("classes")
        if (
            n <= 0
            or f < 0
            or frames <= 0
            or semantic_fragments < 0
            or not isinstance(classes, list)
            or not classes
            or len(set(classes)) != len(classes)
            or any(not isinstance(value, str) or not value for value in classes)
        ):
            return False
        with np.load(directory / "lifting_bank.npz", allow_pickle=False) as a:
            required = {
                "xyz_m", "affinity", "semantic", "label_features",
                "fragment_full_indptr", "fragment_full_ids", "fragment_full_mass",
                "fragment_core_indptr", "fragment_core_ids", "fragment_core_mass",
                "fragment_id", "fragment_frame", "fragment_mask_index",
                "fragment_conflict_ratio", "frame_visible_indptr", "frame_visible_ids",
                "frame_visible_mass", "frame_geometry_abstained",
                "semantic_fragment_full_indptr", "semantic_fragment_full_ids",
                "semantic_fragment_full_mass", "semantic_fragment_frame",
                "semantic_fragment_class",
            }
            if not required.issubset(a.files):
                return False
            xyz = np.asarray(a["xyz_m"])
            affinity = np.asarray(a["affinity"])
            semantic = np.asarray(a["semantic"])
            codebook = np.asarray(a["label_features"])
            if (
                xyz.shape != (n, 3)
                or affinity.ndim != 2
                or affinity.shape[0] != n
                or affinity.shape[1] <= 0
                or semantic.ndim != 2
                or semantic.shape[0] != n
                or semantic.shape[1] <= 0
                or codebook.shape != (len(classes), semantic.shape[1])
                or any(
                    np.any(~np.isfinite(values))
                    for values in (xyz, affinity, semantic, codebook)
                )
                or np.any(np.linalg.norm(codebook, axis=1) <= 0)
                or not np.allclose(
                    np.linalg.norm(codebook, axis=1), 1.0, atol=1e-4, rtol=1e-4
                )
            ):
                return False
            if (
                a["fragment_id"].shape != (f,)
                or not np.issubdtype(a["fragment_id"].dtype, np.integer)
                or len(np.unique(a["fragment_id"])) != f
            ):
                return False
            if (
                a["fragment_conflict_ratio"].shape != (f,)
                or np.any(~np.isfinite(a["fragment_conflict_ratio"]))
                or np.any(a["fragment_conflict_ratio"] < 0)
                or np.any(a["fragment_conflict_ratio"] > 1)
            ):
                return False
            if (
                a["fragment_frame"].shape != (f,)
                or not np.issubdtype(a["fragment_frame"].dtype, np.integer)
                or np.any(a["fragment_frame"] < 0)
                or np.any(a["fragment_frame"] >= frames)
                or a["fragment_mask_index"].shape != (f,)
                or not np.issubdtype(a["fragment_mask_index"].dtype, np.integer)
                or np.any(a["fragment_mask_index"] < 0)
                or a["frame_geometry_abstained"].shape != (frames,)
                or a["frame_geometry_abstained"].dtype != np.bool_
                or "frame_grounded_missing" not in a.files
                or a["frame_grounded_missing"].shape != (frames,)
                or a["frame_grounded_missing"].dtype != np.bool_
            ):
                return False
            fi, fv = a["fragment_full_indptr"], a["fragment_full_ids"]
            ci, cv = a["fragment_core_indptr"], a["fragment_core_ids"]
            if not _valid_ragged(fi, fv, f, n) or not _valid_ragged(ci, cv, f, n):
                return False
            full_mass = np.asarray(a["fragment_full_mass"])
            core_mass = np.asarray(a["fragment_core_mass"])
            if (
                full_mass.shape != fv.shape
                or core_mass.shape != cv.shape
                or np.any(~np.isfinite(full_mass))
                or np.any(~np.isfinite(core_mass))
                or np.any(full_mass < 0)
                or np.any(core_mass < 0)
            ):
                return False
            for index in range(f):
                full = fv[int(fi[index]):int(fi[index + 1])]
                core = cv[int(ci[index]):int(ci[index + 1])]
                if (
                    (len(full) and np.any(np.diff(full) <= 0))
                    or (len(core) and np.any(np.diff(core) <= 0))
                    or not np.all(np.isin(core, full))
                ):
                    return False
            vi, vv = a["frame_visible_indptr"], a["frame_visible_ids"]
            visible_mass = np.asarray(a["frame_visible_mass"])
            if (
                not _valid_ragged(vi, vv, frames, n)
                or visible_mass.shape != vv.shape
                or np.any(~np.isfinite(visible_mass))
                or np.any(visible_mass < 0)
            ):
                return False
            semantic_count = len(a["semantic_fragment_frame"])
            si, sv = a["semantic_fragment_full_indptr"], a["semantic_fragment_full_ids"]
            semantic_mass = np.asarray(a["semantic_fragment_full_mass"])
            semantic_frame = np.asarray(a["semantic_fragment_frame"])
            semantic_class = np.asarray(a["semantic_fragment_class"])
            return bool(
                semantic_count == semantic_fragments
                and _valid_ragged(si, sv, semantic_count, n)
                and semantic_mass.shape == sv.shape
                and np.all(np.isfinite(semantic_mass))
                and np.all(semantic_mass >= 0)
                and semantic_frame.shape == (semantic_count,)
                and np.issubdtype(semantic_frame.dtype, np.integer)
                and np.all(semantic_frame >= 0)
                and np.all(semantic_frame < frames)
                and semantic_class.shape == (semantic_count,)
                and np.issubdtype(semantic_class.dtype, np.integer)
                and np.all(semantic_class >= 0)
                and np.all(semantic_class < len(classes))
            )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def load_lifting_bank(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not lifting_bank_is_complete(directory):
        raise ValueError(f"invalid native V9 lifting bank: {directory}")
    metadata = json.loads((directory / "lifting_bank.json").read_text("utf-8"))
    with np.load(directory / "lifting_bank.npz", allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    return metadata, arrays
