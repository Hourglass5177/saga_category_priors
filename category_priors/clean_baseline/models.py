from __future__ import annotations

"""Data contracts for the clean alpha-mask evidence layer.

The classes in this module contain no renderer, clustering, semantic routing,
or evaluation code.  They only represent one immutable observation bank:
complete 2D masks, their sparse Gaussian support, and per-frame visibility.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Sequence

import numpy as np


EVIDENCE_SCHEMA = "saga-clean-alpha-mask-evidence-v1"
DIAGNOSTICS_SCHEMA = "saga-clean-alpha-mask-evidence-diagnostics-v1"


def _readonly(array: Any, dtype: Any, *, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(array, dtype=dtype)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    result = np.ascontiguousarray(result)
    result.setflags(write=False)
    return result


def _json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = {} if value is None else dict(value)
    try:
        # Round-trip to reject ndarray/path/custom-object values and NaN/Inf.
        encoded = json.dumps(result, allow_nan=False, sort_keys=True)
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("source metadata must be finite JSON data") from exc
    if not isinstance(normalized, dict):  # pragma: no cover - defensive
        raise ValueError("source metadata must be a JSON object")
    return normalized


@dataclass(frozen=True)
class EvidenceThresholds:
    """Frozen membership thresholds used by every clean-baseline condition."""

    visible_min_mass: float = 0.5
    inside_min_mass: float = 0.5
    inside_min_ratio: float = 0.5

    def __post_init__(self) -> None:
        values = (
            float(self.visible_min_mass),
            float(self.inside_min_mass),
            float(self.inside_min_ratio),
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("evidence thresholds must be finite")
        if values[0] <= 0 or values[1] <= 0:
            raise ValueError("mass thresholds must be positive")
        if not 0 < values[2] <= 1:
            raise ValueError("inside_min_ratio must be in (0, 1]")
        object.__setattr__(self, "visible_min_mass", values[0])
        object.__setattr__(self, "inside_min_mass", values[1])
        object.__setattr__(self, "inside_min_ratio", values[2])

    def to_dict(self) -> dict[str, float]:
        return {
            "visible_min_mass": self.visible_min_mass,
            "inside_min_mass": self.inside_min_mass,
            "inside_min_ratio": self.inside_min_ratio,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceThresholds":
        required = {"visible_min_mass", "inside_min_mass", "inside_min_ratio"}
        if set(value) != required:
            raise ValueError("unexpected evidence-threshold fields")
        return cls(**{key: float(value[key]) for key in required})


@dataclass(frozen=True)
class AlphaMassFrame:
    """Dense reference accumulation before thresholding and sparse packing."""

    inside_mass: np.ndarray
    visible_mass: np.ndarray
    valid_pixel_count: int
    geometry_abstained: bool = False

    def __post_init__(self) -> None:
        inside = _readonly(
            self.inside_mass, np.float64, ndim=2, name="inside_mass"
        )
        visible = _readonly(
            self.visible_mass, np.float64, ndim=1, name="visible_mass"
        )
        if inside.shape[1] != len(visible):
            raise ValueError("inside_mass must be mask_count x point_count")
        if np.any(~np.isfinite(inside)) or np.any(~np.isfinite(visible)):
            raise ValueError("alpha masses must be finite")
        if np.any(inside < 0) or np.any(visible < 0):
            raise ValueError("alpha masses must be non-negative")
        tolerance = 5e-6 * np.maximum(visible[None, :], 1.0)
        if inside.size and np.any(inside - visible[None, :] > tolerance):
            raise ValueError("inside mass cannot exceed visible mass")
        pixel_count = int(self.valid_pixel_count)
        if pixel_count < 0:
            raise ValueError("valid_pixel_count must be non-negative")
        if bool(self.geometry_abstained) and inside.shape[0] != 0:
            raise ValueError("an abstained geometry frame cannot contain masks")
        object.__setattr__(self, "inside_mass", inside)
        object.__setattr__(self, "visible_mass", visible)
        object.__setattr__(self, "valid_pixel_count", pixel_count)
        object.__setattr__(self, "geometry_abstained", bool(self.geometry_abstained))

    @property
    def mask_count(self) -> int:
        return int(self.inside_mass.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.visible_mass.shape[0])


@dataclass(frozen=True)
class FrameMetadata:
    frame_id: int
    image_name: str
    valid_pixel_count: int
    geometry_abstained: bool
    semantic_abstained: bool

    def __post_init__(self) -> None:
        if int(self.frame_id) < 0:
            raise ValueError("frame_id must be non-negative")
        if not isinstance(self.image_name, str) or not self.image_name.strip():
            raise ValueError("image_name must be a non-empty string")
        if int(self.valid_pixel_count) < 0:
            raise ValueError("valid_pixel_count must be non-negative")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "image_name", self.image_name.strip())
        object.__setattr__(self, "valid_pixel_count", int(self.valid_pixel_count))
        object.__setattr__(self, "geometry_abstained", bool(self.geometry_abstained))
        object.__setattr__(self, "semantic_abstained", bool(self.semantic_abstained))


@dataclass(frozen=True)
class MaskMetadata:
    global_mask_id: int
    frame_id: int
    image_name: str
    mask_index: int

    def __post_init__(self) -> None:
        if int(self.global_mask_id) < 0:
            raise ValueError("global_mask_id must be non-negative")
        if int(self.frame_id) < 0 or int(self.mask_index) < 0:
            raise ValueError("frame_id and mask_index must be non-negative")
        if not isinstance(self.image_name, str) or not self.image_name.strip():
            raise ValueError("mask image_name must be non-empty")
        object.__setattr__(self, "global_mask_id", int(self.global_mask_id))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "mask_index", int(self.mask_index))
        object.__setattr__(self, "image_name", self.image_name.strip())


def _validate_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    *,
    row_count: int,
    column_count: int,
    name: str,
) -> None:
    if indptr.shape != (int(row_count) + 1,):
        raise ValueError(f"{name}.indptr has the wrong length")
    if int(indptr[0]) != 0 or np.any(np.diff(indptr) < 0):
        raise ValueError(f"{name}.indptr is not monotonic CSR")
    if int(indptr[-1]) != len(indices):
        raise ValueError(f"{name}.indptr does not terminate at nnz")
    if np.any(indices < 0) or np.any(indices >= int(column_count)):
        raise ValueError(f"{name}.indices are out of range")
    for row in range(int(row_count)):
        values = indices[int(indptr[row]) : int(indptr[row + 1])]
        if len(values) and np.any(np.diff(values) <= 0):
            raise ValueError(f"{name} row {row} must be sorted and unique")


@dataclass(frozen=True)
class MaskSupportCSR:
    """Mask rows by Gaussian columns, including same-frame ambiguity flags."""

    indptr: np.ndarray
    gaussian_ids: np.ndarray
    inside_mass: np.ndarray
    inside_ratio: np.ndarray
    ambiguous: np.ndarray
    row_count: int
    point_count: int

    def __post_init__(self) -> None:
        indptr = _readonly(self.indptr, np.int64, ndim=1, name="support.indptr")
        ids = _readonly(
            self.gaussian_ids, np.int32, ndim=1, name="support.gaussian_ids"
        )
        mass = _readonly(
            self.inside_mass, np.float32, ndim=1, name="support.inside_mass"
        )
        ratio = _readonly(
            self.inside_ratio, np.float32, ndim=1, name="support.inside_ratio"
        )
        ambiguous = _readonly(
            self.ambiguous, np.bool_, ndim=1, name="support.ambiguous"
        )
        rows, points = int(self.row_count), int(self.point_count)
        if rows < 0 or points <= 0:
            raise ValueError("support row_count must be non-negative and point_count positive")
        _validate_csr(
            indptr, ids, row_count=rows, column_count=points, name="mask_support"
        )
        if mass.shape != ids.shape or ratio.shape != ids.shape or ambiguous.shape != ids.shape:
            raise ValueError("support data arrays must all have nnz entries")
        if (
            np.any(~np.isfinite(mass))
            or np.any(~np.isfinite(ratio))
            or np.any(mass < 0)
            or np.any(ratio < 0)
            or np.any(ratio > 1 + 1e-6)
        ):
            raise ValueError("support mass/ratio values are invalid")
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "gaussian_ids", ids)
        object.__setattr__(self, "inside_mass", mass)
        object.__setattr__(self, "inside_ratio", ratio)
        object.__setattr__(self, "ambiguous", ambiguous)
        object.__setattr__(self, "row_count", rows)
        object.__setattr__(self, "point_count", points)

    def row(
        self, index: int, *, include_ambiguous: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        row = int(index)
        if not 0 <= row < self.row_count:
            raise IndexError("mask-support row is out of range")
        start, stop = int(self.indptr[row]), int(self.indptr[row + 1])
        selected = np.ones(stop - start, dtype=bool)
        if not include_ambiguous:
            selected &= ~self.ambiguous[start:stop]
        return (
            self.gaussian_ids[start:stop][selected],
            self.inside_mass[start:stop][selected],
            self.inside_ratio[start:stop][selected],
            self.ambiguous[start:stop][selected],
        )


@dataclass(frozen=True)
class PackedVisibility:
    """Frame rows by visible Gaussian columns."""

    indptr: np.ndarray
    gaussian_ids: np.ndarray
    visible_mass: np.ndarray
    row_count: int
    point_count: int

    def __post_init__(self) -> None:
        indptr = _readonly(self.indptr, np.int64, ndim=1, name="visibility.indptr")
        ids = _readonly(
            self.gaussian_ids, np.int32, ndim=1, name="visibility.gaussian_ids"
        )
        mass = _readonly(
            self.visible_mass, np.float32, ndim=1, name="visibility.visible_mass"
        )
        rows, points = int(self.row_count), int(self.point_count)
        if rows < 0 or points <= 0:
            raise ValueError("visibility row_count must be non-negative and point_count positive")
        _validate_csr(
            indptr, ids, row_count=rows, column_count=points, name="frame_visibility"
        )
        if mass.shape != ids.shape or np.any(~np.isfinite(mass)) or np.any(mass < 0):
            raise ValueError("visible_mass must be finite, non-negative, and match nnz")
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "gaussian_ids", ids)
        object.__setattr__(self, "visible_mass", mass)
        object.__setattr__(self, "row_count", rows)
        object.__setattr__(self, "point_count", points)

    def row(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        row = int(index)
        if not 0 <= row < self.row_count:
            raise IndexError("visibility row is out of range")
        start, stop = int(self.indptr[row]), int(self.indptr[row + 1])
        return self.gaussian_ids[start:stop], self.visible_mass[start:stop]


@dataclass(frozen=True)
class PackedIndexRows:
    """CSR-like rows containing only sorted integer IDs."""

    indptr: np.ndarray
    ids: np.ndarray
    row_count: int
    upper_bound: int
    name: str = "packed_indices"

    def __post_init__(self) -> None:
        indptr = _readonly(self.indptr, np.int64, ndim=1, name=f"{self.name}.indptr")
        ids = _readonly(self.ids, np.int32, ndim=1, name=f"{self.name}.ids")
        rows, upper = int(self.row_count), int(self.upper_bound)
        if rows < 0 or upper <= 0:
            raise ValueError("packed rows require non-negative rows and positive upper bound")
        _validate_csr(indptr, ids, row_count=rows, column_count=upper, name=self.name)
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "row_count", rows)
        object.__setattr__(self, "upper_bound", upper)

    def row(self, index: int) -> np.ndarray:
        row = int(index)
        if not 0 <= row < self.row_count:
            raise IndexError(f"{self.name} row is out of range")
        return self.ids[int(self.indptr[row]) : int(self.indptr[row + 1])]


@dataclass(frozen=True)
class FrameEvidence:
    """One frame before packing into a scene-level bank."""

    metadata: FrameMetadata
    masks: tuple[MaskMetadata, ...]
    support: MaskSupportCSR
    visibility: PackedVisibility
    ambiguous_gaussians: np.ndarray
    semantic_posteriors: np.ndarray
    semantic_abstained: np.ndarray

    def __post_init__(self) -> None:
        masks = tuple(self.masks)
        if self.support.row_count != len(masks) or self.visibility.row_count != 1:
            raise ValueError("frame support/visibility row counts are inconsistent")
        if self.support.point_count != self.visibility.point_count:
            raise ValueError("frame support and visibility point counts differ")
        ambiguous = _readonly(
            self.ambiguous_gaussians,
            np.int32,
            ndim=1,
            name="ambiguous_gaussians",
        )
        if (
            np.any(ambiguous < 0)
            or np.any(ambiguous >= self.support.point_count)
            or (len(ambiguous) and np.any(np.diff(ambiguous) <= 0))
        ):
            raise ValueError("ambiguous_gaussians must be sorted unique point IDs")
        posterior = _readonly(
            self.semantic_posteriors,
            np.float32,
            ndim=2,
            name="semantic_posteriors",
        )
        abstained = _readonly(
            self.semantic_abstained,
            np.bool_,
            ndim=1,
            name="semantic_abstained",
        )
        if posterior.shape[0] != len(masks) or abstained.shape != (len(masks),):
            raise ValueError("semantic rows must match masks")
        if np.any(~np.isfinite(posterior)) or np.any(posterior < 0):
            raise ValueError("semantic posteriors must be finite and non-negative")
        row_sum = posterior.sum(axis=1, dtype=np.float64)
        if np.any(abstained & (row_sum > 1e-7)):
            raise ValueError("abstained semantic rows must be all zero")
        if np.any(~abstained & ~np.isclose(row_sum, 1.0, atol=1e-5, rtol=1e-5)):
            raise ValueError("non-abstained semantic rows must sum to one")
        local = [mask.mask_index for mask in masks]
        global_ids = [mask.global_mask_id for mask in masks]
        if local != sorted(local) or len(local) != len(set(local)):
            raise ValueError("frame mask_index values must be sorted and unique")
        if len(global_ids) != len(set(global_ids)):
            raise ValueError("frame global_mask_id values must be unique")
        for mask in masks:
            if mask.frame_id != self.metadata.frame_id or mask.image_name != self.metadata.image_name:
                raise ValueError("mask metadata does not belong to its frame")
        if self.metadata.geometry_abstained and masks:
            raise ValueError("an abstained geometry frame cannot contain masks")
        expected_semantic_abstention = bool(len(masks) == 0 or np.all(abstained))
        if self.metadata.semantic_abstained != expected_semantic_abstention:
            raise ValueError("frame semantic abstention does not match mask rows")
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "ambiguous_gaussians", ambiguous)
        object.__setattr__(self, "semantic_posteriors", posterior)
        object.__setattr__(self, "semantic_abstained", abstained)


@dataclass(frozen=True)
class AlphaMaskEvidenceBank:
    """Immutable, scene-level sparse evidence consumed by consensus methods."""

    scene_id: str
    point_count: int
    xyz_m: np.ndarray
    class_names: tuple[str, ...]
    thresholds: EvidenceThresholds
    frames: tuple[FrameMetadata, ...]
    masks: tuple[MaskMetadata, ...]
    mask_support: MaskSupportCSR
    frame_visibility: PackedVisibility
    frame_ambiguity: PackedIndexRows
    semantic_posteriors: np.ndarray
    semantic_abstained: np.ndarray
    source: Mapping[str, Any] = field(default_factory=dict)
    schema: str = EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        scene = str(self.scene_id).strip()
        points = int(self.point_count)
        classes = tuple(str(value).strip() for value in self.class_names)
        frames = tuple(self.frames)
        masks = tuple(self.masks)
        if self.schema != EVIDENCE_SCHEMA:
            raise ValueError(f"unsupported evidence schema: {self.schema!r}")
        if not scene or points <= 0:
            raise ValueError("scene_id must be non-empty and point_count positive")
        xyz = _readonly(self.xyz_m, np.float32, ndim=2, name="xyz_m")
        if xyz.shape != (points, 3) or np.any(~np.isfinite(xyz)):
            raise ValueError("xyz_m must be a finite point_count x 3 metric array")
        if (
            len(classes) != 32
            or any(not value for value in classes)
            or len(set(classes)) != len(classes)
        ):
            raise ValueError("class_names must contain exactly 32 unique names")
        if not frames:
            raise ValueError("an evidence bank must contain at least one frame")
        frame_ids = [frame.frame_id for frame in frames]
        if frame_ids != sorted(frame_ids) or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("frames must be sorted by unique frame_id")
        global_ids = [mask.global_mask_id for mask in masks]
        if len(global_ids) != len(set(global_ids)):
            raise ValueError("global_mask_id values must be unique across the bank")
        frame_by_id = {frame.frame_id: frame for frame in frames}
        for mask in masks:
            frame = frame_by_id.get(mask.frame_id)
            if frame is None or frame.image_name != mask.image_name:
                raise ValueError("mask references an unknown or different frame")
        if (
            self.mask_support.row_count != len(masks)
            or self.mask_support.point_count != points
            or self.frame_visibility.row_count != len(frames)
            or self.frame_visibility.point_count != points
            or self.frame_ambiguity.row_count != len(frames)
            or self.frame_ambiguity.upper_bound != points
        ):
            raise ValueError("packed evidence dimensions do not match bank metadata")
        posterior = _readonly(
            self.semantic_posteriors,
            np.float32,
            ndim=2,
            name="bank.semantic_posteriors",
        )
        abstained = _readonly(
            self.semantic_abstained,
            np.bool_,
            ndim=1,
            name="bank.semantic_abstained",
        )
        if posterior.shape != (len(masks), len(classes)) or abstained.shape != (len(masks),):
            raise ValueError("bank semantic posterior dimensions are inconsistent")
        if np.any(~np.isfinite(posterior)) or np.any(posterior < 0):
            raise ValueError("bank semantic posteriors must be finite and non-negative")
        sums = posterior.sum(axis=1, dtype=np.float64)
        if np.any(abstained & (sums > 1e-7)) or np.any(
            ~abstained & ~np.isclose(sums, 1.0, atol=1e-5, rtol=1e-5)
        ):
            raise ValueError("bank semantic abstention/posterior contract is invalid")
        source = _json_mapping(self.source)
        object.__setattr__(self, "scene_id", scene)
        object.__setattr__(self, "point_count", points)
        object.__setattr__(self, "xyz_m", xyz)
        object.__setattr__(self, "class_names", classes)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "semantic_posteriors", posterior)
        object.__setattr__(self, "semantic_abstained", abstained)
        object.__setattr__(self, "source", source)
        self._validate_cross_arrays()

    def _validate_cross_arrays(self) -> None:
        frame_position = {frame.frame_id: index for index, frame in enumerate(self.frames)}
        masks_by_frame: dict[int, list[int]] = {frame.frame_id: [] for frame in self.frames}
        for mask_row, mask in enumerate(self.masks):
            masks_by_frame[mask.frame_id].append(mask_row)
            visible_ids, visible_mass = self.frame_visibility.row(frame_position[mask.frame_id])
            support_ids, inside, ratios, flags = self.mask_support.row(mask_row)
            positions = np.searchsorted(visible_ids, support_ids)
            if len(support_ids) and (
                np.any(positions >= len(visible_ids))
                or np.any(visible_ids[positions] != support_ids)
            ):
                raise ValueError("mask support must be a subset of frame visibility")
            if len(support_ids):
                visible = visible_mass[positions]
                if np.any(inside - visible > 5e-5 * np.maximum(visible, 1.0)):
                    raise ValueError("mask inside mass exceeds packed visible mass")
                expected = inside / visible
                if not np.allclose(ratios, expected, atol=2e-5, rtol=2e-5):
                    raise ValueError("mask inside ratios disagree with mass/visibility")
            ambiguous_ids = self.frame_ambiguity.row(frame_position[mask.frame_id])
            if not np.array_equal(flags, np.isin(support_ids, ambiguous_ids)):
                raise ValueError("support ambiguity flags disagree with frame ambiguity")
        for frame in self.frames:
            rows = masks_by_frame[frame.frame_id]
            ambiguous = self.frame_ambiguity.row(frame_position[frame.frame_id])
            counts: dict[int, int] = {}
            for row in rows:
                ids = self.mask_support.row(row)[0]
                for value in ids.tolist():
                    counts[value] = counts.get(value, 0) + 1
            expected = np.asarray(
                sorted(value for value, count in counts.items() if count > 1),
                dtype=np.int32,
            )
            if not np.array_equal(ambiguous, expected):
                raise ValueError("frame ambiguity must be exactly multi-mask support")
            row_abstained = self.semantic_abstained[rows]
            expected_semantic = bool(len(rows) == 0 or np.all(row_abstained))
            if frame.semantic_abstained != expected_semantic:
                raise ValueError("frame semantic abstention is inconsistent")

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def mask_count(self) -> int:
        return len(self.masks)

    def frame_position(self, frame_id: int) -> int:
        target = int(frame_id)
        for index, frame in enumerate(self.frames):
            if frame.frame_id == target:
                return index
        raise KeyError(f"unknown frame_id: {target}")

    def mask_position(self, global_mask_id: int) -> int:
        target = int(global_mask_id)
        for index, mask in enumerate(self.masks):
            if mask.global_mask_id == target:
                return index
        raise KeyError(f"unknown global_mask_id: {target}")

    def support_for_mask(
        self, global_mask_id: int, *, include_ambiguous: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.mask_support.row(
            self.mask_position(global_mask_id),
            include_ambiguous=include_ambiguous,
        )

    def visibility_for_frame(self, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        return self.frame_visibility.row(self.frame_position(frame_id))

    def ambiguous_for_frame(self, frame_id: int) -> np.ndarray:
        return self.frame_ambiguity.row(self.frame_position(frame_id))

    def masks_for_gaussian(
        self, gaussian_id: int, *, include_ambiguous: bool = False
    ) -> np.ndarray:
        point = int(gaussian_id)
        if not 0 <= point < self.point_count:
            raise IndexError("gaussian_id is out of range")
        found: list[int] = []
        for row, metadata in enumerate(self.masks):
            ids, _, _, _ = self.mask_support.row(
                row, include_ambiguous=include_ambiguous
            )
            position = int(np.searchsorted(ids, point))
            if position < len(ids) and int(ids[position]) == point:
                found.append(metadata.global_mask_id)
        output = np.asarray(found, dtype=np.int64)
        output.setflags(write=False)
        return output

    def frames_for_gaussian(
        self, gaussian_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return stable frame IDs and visible mass for one Gaussian.

        The bank is stored frame-major because consensus operates on complete
        mask observations.  This inverse query deliberately returns the
        physical ``frame_id`` values, rather than packed row positions, so a
        caller cannot accidentally confuse the two coordinate systems.
        """

        point = int(gaussian_id)
        if not 0 <= point < self.point_count:
            raise IndexError("gaussian_id is out of range")
        frame_ids: list[int] = []
        masses: list[float] = []
        for row, metadata in enumerate(self.frames):
            ids, values = self.frame_visibility.row(row)
            position = int(np.searchsorted(ids, point))
            if position < len(ids) and int(ids[position]) == point:
                frame_ids.append(metadata.frame_id)
                masses.append(float(values[position]))
        output_ids = np.asarray(frame_ids, dtype=np.int64)
        output_mass = np.asarray(masses, dtype=np.float32)
        output_ids.setflags(write=False)
        output_mass.setflags(write=False)
        return output_ids, output_mass

    def save(self, directory: Any, *, overwrite: bool = False) -> None:
        """Persist this bank through the strict evidence serializer."""

        # Local import keeps the data-contract module independent of the
        # filesystem implementation while avoiding a module import cycle.
        from .evidence import save_evidence_bank

        save_evidence_bank(self, directory, overwrite=overwrite)

    @classmethod
    def from_frames(
        cls,
        *,
        scene_id: str,
        point_count: int,
        xyz_m: np.ndarray,
        class_names: Sequence[str],
        frames: Sequence[FrameEvidence],
        thresholds: EvidenceThresholds = EvidenceThresholds(),
        source: Mapping[str, Any] | None = None,
    ) -> "AlphaMaskEvidenceBank":
        ordered = tuple(sorted(frames, key=lambda value: value.metadata.frame_id))
        points = int(point_count)
        classes = tuple(class_names)
        if any(frame.support.point_count != points for frame in ordered):
            raise ValueError("all frames must use the bank point_count")
        if any(frame.semantic_posteriors.shape[1] != len(classes) for frame in ordered):
            raise ValueError("all frames must use the bank class_names")
        frame_meta = tuple(frame.metadata for frame in ordered)
        mask_meta = tuple(mask for frame in ordered for mask in frame.masks)

        support_lengths = [
            int(frame.support.indptr[row + 1] - frame.support.indptr[row])
            for frame in ordered
            for row in range(frame.support.row_count)
        ]
        support_indptr = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(support_lengths, dtype=np.int64))
        )
        def concatenate(values: list[np.ndarray], dtype: Any) -> np.ndarray:
            return (
                np.concatenate([np.asarray(value, dtype=dtype) for value in values])
                if values and sum(len(value) for value in values)
                else np.empty(0, dtype=dtype)
            )

        support_rows = [
            frame.support.row(row)
            for frame in ordered
            for row in range(frame.support.row_count)
        ]
        support = MaskSupportCSR(
            support_indptr,
            concatenate([row[0] for row in support_rows], np.int32),
            concatenate([row[1] for row in support_rows], np.float32),
            concatenate([row[2] for row in support_rows], np.float32),
            concatenate([row[3] for row in support_rows], np.bool_),
            len(mask_meta),
            points,
        )
        visible_rows = [frame.visibility.row(0) for frame in ordered]
        visible_lengths = [len(row[0]) for row in visible_rows]
        visibility = PackedVisibility(
            np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(visible_lengths, dtype=np.int64))
            ),
            concatenate([row[0] for row in visible_rows], np.int32),
            concatenate([row[1] for row in visible_rows], np.float32),
            len(ordered),
            points,
        )
        ambiguity_lengths = [len(frame.ambiguous_gaussians) for frame in ordered]
        ambiguity = PackedIndexRows(
            np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(ambiguity_lengths, dtype=np.int64))
            ),
            concatenate([frame.ambiguous_gaussians for frame in ordered], np.int32),
            len(ordered),
            points,
            "frame_ambiguity",
        )
        posterior = (
            np.concatenate([frame.semantic_posteriors for frame in ordered], axis=0)
            if ordered
            else np.empty((0, len(classes)), dtype=np.float32)
        )
        abstained = concatenate(
            [frame.semantic_abstained for frame in ordered], np.bool_
        )
        return cls(
            scene_id=str(scene_id),
            point_count=points,
            xyz_m=xyz_m,
            class_names=classes,
            thresholds=thresholds,
            frames=frame_meta,
            masks=mask_meta,
            mask_support=support,
            frame_visibility=visibility,
            frame_ambiguity=ambiguity,
            semantic_posteriors=posterior,
            semantic_abstained=abstained,
            source={} if source is None else source,
        )
