from __future__ import annotations

"""Frozen configuration and artifact contracts for the teacher baseline closeout.

This module contains no evaluator and launches no subprocesses.  In particular,
it never checks out or patches a source tree: every historical source variant is
supplied as an already prepared, isolated workspace to the runner.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BFC18_CLASSES = (
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
)
BFC18_SELECTED_CLASSES = (
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
    "socket",
    "book",
    "remote",
    "key",
)
BFC18_OTHER_CLASSES = ("socket", "book", "remote", "key")

TIP28_CLASSES = BFC18_CLASSES + (
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
)
TIP28_SELECTED_CLASSES = tuple(
    value
    for value in TIP28_CLASSES
    if value not in {"wall", "floor", "ceiling", "person"}
)
TIP28_OTHER_CLASSES = BFC18_OTHER_CLASSES + ("cup", "vase", "phone")

CLOSURE_SCENES = ("scene0064_01", "scene0025_01", "scene0231_00")
PRIMARY_COMMIT = "bfc21922384cc991a71b5e51429354b5d6b06375"
FULL950_COMMIT = "95073c640a77984c6af24abb276147e4315abcd1"
EVOLUTION_COMMIT = "8c5e167493b26987c9c52e2e05caf0c6d7406789"
# Recovered from an otherwise unreachable ``git stash`` merge commit.  Its
# first parent is exactly EVOLUTION_COMMIT and its tree contains the tracked
# dirty worktree (the third parent contains the untracked ``.codex`` file).
# This is a byte-recoverable handoff candidate, but the historical office
# output remains the behavioral oracle because the downloaded archive itself
# was never checksummed.
RECOVERED_DIRTY_COMMIT = "5804fcb2243e165197ac305b286ac34bd4fdaf68"


@dataclass(frozen=True)
class TaxonomySpec:
    taxonomy_id: str
    classes: tuple[str, ...]
    selected_classes: tuple[str, ...]
    other_classes: tuple[str, ...]
    semantic_feature_shape: tuple[int, int]


@dataclass(frozen=True)
class SourceVariantSpec:
    variant_id: str
    base_commit: str
    patch_set: str | None
    exact_commit: str | None
    has_args_plumbing: bool
    normalized_feature_dim: bool
    sorted_semantic_masks: bool
    integer_iteration_cli: bool = False


@dataclass(frozen=True)
class RunSpec:
    variant_id: str
    budget: str
    scene_ids: tuple[str, ...]
    conditions: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeScene:
    scene_id: str
    base_path: Path
    python_bin: Path

    @property
    def images_path(self) -> Path:
        return self.base_path / "fastRecon/dense/sparse/0/images"

    @property
    def sparse_path(self) -> Path:
        return self.base_path / "fastRecon/dense/sparse/0"

    @property
    def point_cloud_path(self) -> Path:
        root = self.base_path / "output_models/point_cloud/iteration_30000"
        teacher_name = root / "scene_point_cloud.ply"
        generic_name = root / "point_cloud.ply"
        if teacher_name.is_file():
            return teacher_name
        if generic_name.is_file():
            return generic_name
        # Preserve the teacher handoff name in the eventual preflight error.
        return teacher_name


@dataclass(frozen=True)
class SourceWorkspace:
    variant_id: str
    root: Path


@dataclass(frozen=True)
class AssetPaths:
    root: Path
    masks: Path
    labels: Path
    label_features: Path
    mask_scales: Path
    masks_summary: Path
    masks_progress: Path
    scale_progress: Path


@dataclass(frozen=True)
class FeaturePaths:
    root: Path
    point_cloud: Path
    scale_gate: Path
    progress: Path


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    output_json: Path
    progress: Path


TAXONOMIES: Mapping[str, TaxonomySpec] = {
    "bfc18": TaxonomySpec(
        taxonomy_id="bfc18",
        classes=BFC18_CLASSES,
        selected_classes=BFC18_SELECTED_CLASSES,
        other_classes=BFC18_OTHER_CLASSES,
        semantic_feature_shape=(18, 32),
    ),
    "tip28": TaxonomySpec(
        taxonomy_id="tip28",
        classes=TIP28_CLASSES,
        selected_classes=TIP28_SELECTED_CLASSES,
        other_classes=TIP28_OTHER_CLASSES,
        semantic_feature_shape=(28, 32),
    ),
}

SOURCE_VARIANTS: Mapping[str, SourceVariantSpec] = {
    "literal-bfc": SourceVariantSpec(
        variant_id="literal-bfc",
        base_commit=PRIMARY_COMMIT,
        patch_set=None,
        exact_commit=PRIMARY_COMMIT,
        has_args_plumbing=False,
        normalized_feature_dim=False,
        sorted_semantic_masks=False,
    ),
    "args-only": SourceVariantSpec(
        variant_id="args-only",
        base_commit=PRIMARY_COMMIT,
        patch_set="args-only",
        exact_commit=None,
        has_args_plumbing=True,
        normalized_feature_dim=False,
        sorted_semantic_masks=False,
    ),
    "args-norm": SourceVariantSpec(
        variant_id="args-norm",
        base_commit=PRIMARY_COMMIT,
        patch_set="args-norm",
        exact_commit=None,
        has_args_plumbing=True,
        normalized_feature_dim=True,
        sorted_semantic_masks=False,
    ),
    "full950": SourceVariantSpec(
        variant_id="full950",
        base_commit=PRIMARY_COMMIT,
        patch_set="full950",
        exact_commit=FULL950_COMMIT,
        has_args_plumbing=True,
        normalized_feature_dim=True,
        sorted_semantic_masks=True,
    ),
    "full950-iterations-cli": SourceVariantSpec(
        variant_id="full950-iterations-cli",
        base_commit=FULL950_COMMIT,
        patch_set="iterations-none-to-zero",
        exact_commit=None,
        has_args_plumbing=True,
        normalized_feature_dim=True,
        sorted_semantic_masks=True,
        integer_iteration_cli=True,
    ),
    "tip8c": SourceVariantSpec(
        variant_id="tip8c",
        base_commit=PRIMARY_COMMIT,
        patch_set="tip8c",
        exact_commit=EVOLUTION_COMMIT,
        has_args_plumbing=True,
        normalized_feature_dim=True,
        sorted_semantic_masks=True,
    ),
    "tip8c-dirty-recovered": SourceVariantSpec(
        variant_id="tip8c-dirty-recovered",
        base_commit=EVOLUTION_COMMIT,
        patch_set="recovered-stash-worktree",
        exact_commit=RECOVERED_DIRTY_COMMIT,
        has_args_plumbing=True,
        normalized_feature_dim=True,
        sorted_semantic_masks=True,
    ),
}

# The 8c source tip is deliberately absent from the prototype causal matrix.
# It is the likely delivered code candidate and is handled by the V9 handoff
# recovery/office behavior oracle rather than being mislabeled as exact here.
REGISTERED_RUNS = (
    RunSpec(
        variant_id="literal-bfc",
        budget="adaptive",
        scene_ids=("scene0064_01",),
        conditions=("B0-global", "B1-original"),
    ),
    RunSpec(
        variant_id="args-only",
        budget="adaptive",
        scene_ids=("scene0064_01",),
        conditions=("B0-global", "B1-original"),
    ),
    RunSpec(
        variant_id="args-norm",
        budget="adaptive",
        scene_ids=("scene0064_01",),
        conditions=("B0-global", "B1-original"),
    ),
    RunSpec(
        variant_id="full950",
        budget="adaptive",
        scene_ids=CLOSURE_SCENES,
        conditions=("B0-global", "B1-original"),
    ),
    RunSpec(
        variant_id="full950-iterations-cli",
        budget="adaptive",
        scene_ids=("scene0064_01",),
        conditions=("B0-global", "B1-original"),
    ),
    RunSpec(
        variant_id="full950-iterations-cli",
        budget="10000",
        scene_ids=("scene0064_01",),
        conditions=("B0-global", "B1-original"),
    ),
)


def load_runtime_scenes(path: Path) -> dict[str, RuntimeScene]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: Any = (
        payload.get("scenes", payload) if isinstance(payload, Mapping) else payload
    )
    if isinstance(rows, Mapping):
        rows = [dict(value, scene_id=key) for key, value in rows.items()]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("runtime manifest must contain a scene list or mapping")
    result: dict[str, RuntimeScene] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("runtime scene rows must be objects")
        scene = RuntimeScene(
            scene_id=str(raw["scene_id"]),
            base_path=Path(str(raw["base_path"])),
            python_bin=Path(str(raw["python_bin"])),
        )
        if scene.scene_id in result:
            raise ValueError(f"duplicate runtime scene: {scene.scene_id}")
        result[scene.scene_id] = scene
    return result


def load_source_workspaces(path: Path) -> dict[str, SourceWorkspace]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: Any = (
        payload.get("workspaces", payload) if isinstance(payload, Mapping) else payload
    )
    if isinstance(rows, Mapping):
        rows = [
            {"variant_id": key, "root": value}
            if not isinstance(value, Mapping)
            else dict(value, variant_id=key)
            for key, value in rows.items()
        ]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("workspace manifest must contain a list or mapping")
    result: dict[str, SourceWorkspace] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("workspace rows must be objects")
        workspace = SourceWorkspace(
            variant_id=str(raw["variant_id"]),
            root=Path(str(raw["root"])),
        )
        if workspace.variant_id not in SOURCE_VARIANTS:
            raise ValueError(f"unknown source variant: {workspace.variant_id}")
        if workspace.variant_id in result:
            raise ValueError(f"duplicate source variant: {workspace.variant_id}")
        result[workspace.variant_id] = workspace
    return result


def asset_paths(
    output_root: Path, scene_id: str, taxonomy_id: str = "bfc18"
) -> AssetPaths:
    root = output_root / "assets" / taxonomy_id / scene_id
    return AssetPaths(
        root=root,
        masks=root / "masks",
        labels=root / "labels",
        label_features=root / "labels/label_features.pt",
        mask_scales=root / "mask_scales",
        masks_summary=root / "masks_summary.json",
        masks_progress=root / "progress/masks.txt",
        scale_progress=root / "progress/scale.txt",
    )


def feature_paths(
    output_root: Path,
    scene_id: str,
    variant_id: str,
    budget: str,
    taxonomy_id: str = "bfc18",
) -> FeaturePaths:
    root = output_root / "features" / taxonomy_id / variant_id / budget / scene_id
    return FeaturePaths(
        root=root,
        point_cloud=root / "contrastive_feature_point_cloud.ply",
        scale_gate=root / "scale_gate.pt",
        progress=root / "progress.txt",
    )


def output_paths(
    output_root: Path,
    scene_id: str,
    variant_id: str,
    budget: str,
    condition: str,
    taxonomy_id: str = "bfc18",
) -> OutputPaths:
    root = (
        output_root
        / "outputs"
        / taxonomy_id
        / variant_id
        / budget
        / condition
        / scene_id
    )
    return OutputPaths(
        root=root, output_json=root / "output.json", progress=root / "progress.txt"
    )


def read_ply_vertex_count(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            vertex_count: int | None = None
            for _ in range(4096):
                line = handle.readline()
                if not line:
                    return None
                text = line.decode("ascii", errors="strict").strip()
                if text.startswith("element vertex "):
                    vertex_count = int(text.rsplit(" ", 1)[1])
                if text == "end_header":
                    return vertex_count
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return None


def _progress_complete(path: Path) -> bool:
    try:
        return int(path.read_text(encoding="utf-8").strip()) == 100
    except (OSError, ValueError):
        return False


def _load_torch(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch releases used by the historical workspaces
        return torch.load(path, map_location="cpu")


def _tensor_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None


def _validated_mask_stems(
    scene: RuntimeScene,
    paths: AssetPaths,
    taxonomy: TaxonomySpec,
) -> tuple[set[str], set[str]] | None:
    if not _progress_complete(paths.masks_progress):
        return None
    try:
        images = {
            path.stem
            for path in scene.images_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        }
        mask_files = {path.stem: path for path in paths.masks.glob("*.pt")}
        label_files = {
            path.stem: path
            for path in paths.labels.glob("*.pt")
            if path.name != "label_features.pt"
        }
        if (
            not images
            or set(mask_files) != set(label_files)
            or not set(mask_files) <= images
        ):
            return None
        if (
            _tensor_shape(_load_torch(paths.label_features))
            != taxonomy.semantic_feature_shape
        ):
            return None
        for stem, mask_file in mask_files.items():
            masks = _load_torch(mask_file)
            labels = _load_torch(label_files[stem])
            mask_shape = _tensor_shape(masks)
            label_shape = _tensor_shape(labels)
            if (
                mask_shape is None
                or len(mask_shape) != 3
                or label_shape != (mask_shape[0],)
            ):
                return None
        return images, set(mask_files)
    except (OSError, RuntimeError, ValueError, TypeError, ImportError, KeyError):
        return None


def record_masks_completion(
    scene: RuntimeScene,
    paths: AssetPaths,
    taxonomy: TaxonomySpec,
) -> None:
    stems = _validated_mask_stems(scene, paths, taxonomy)
    if stems is None:
        raise RuntimeError(f"mask stage returned incomplete artifacts: {paths.root}")
    images, outputs = stems
    payload = {
        "schema": "saga-teacher-mask-stage-v1",
        "scene_id": scene.scene_id,
        "taxonomy_id": taxonomy.taxonomy_id,
        "semantic_feature_shape": list(taxonomy.semantic_feature_shape),
        "image_stems": sorted(images),
        "output_stems": sorted(outputs),
    }
    paths.masks_summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def masks_are_complete(
    scene: RuntimeScene, paths: AssetPaths, taxonomy: TaxonomySpec
) -> bool:
    stems = _validated_mask_stems(scene, paths, taxonomy)
    if stems is None or not paths.masks_summary.is_file():
        return False
    images, outputs = stems
    try:
        summary = json.loads(paths.masks_summary.read_text(encoding="utf-8"))
        return bool(
            summary.get("schema") == "saga-teacher-mask-stage-v1"
            and summary.get("scene_id") == scene.scene_id
            and summary.get("taxonomy_id") == taxonomy.taxonomy_id
            and summary.get("semantic_feature_shape")
            == list(taxonomy.semantic_feature_shape)
            and summary.get("image_stems") == sorted(images)
            and summary.get("output_stems") == sorted(outputs)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def scales_are_complete(paths: AssetPaths) -> bool:
    if not _progress_complete(paths.scale_progress):
        return False
    try:
        mask_files = {path.stem: path for path in paths.masks.glob("*.pt")}
        scale_files = {path.stem: path for path in paths.mask_scales.glob("*.pt")}
        if set(scale_files) != set(mask_files):
            return False
        for stem, mask_path in mask_files.items():
            mask_shape = _tensor_shape(_load_torch(mask_path))
            scale_shape = _tensor_shape(_load_torch(scale_files[stem]))
            if mask_shape is None or scale_shape != (mask_shape[0],):
                return False
        return True
    except (OSError, RuntimeError, ValueError, TypeError, ImportError, KeyError):
        return False


def feature_is_complete(scene: RuntimeScene, paths: FeaturePaths) -> bool:
    if not _progress_complete(paths.progress):
        return False
    source_count = read_ply_vertex_count(scene.point_cloud_path)
    feature_count = read_ply_vertex_count(paths.point_cloud)
    if source_count is None or feature_count != source_count or source_count <= 0:
        return False
    try:
        state = _load_torch(paths.scale_gate)
        if not isinstance(state, Mapping):
            return False
        weight = state.get("0.weight")
        bias = state.get("0.bias")
        return _tensor_shape(weight) == (32, 1) and _tensor_shape(bias) == (32,)
    except (OSError, RuntimeError, ValueError, TypeError, ImportError):
        return False


def output_is_complete(
    scene: RuntimeScene,
    paths: OutputPaths,
    taxonomy: TaxonomySpec,
) -> bool:
    point_count = read_ply_vertex_count(scene.point_cloud_path)
    if point_count is None or not paths.output_json.is_file():
        return False
    try:
        payload = json.loads(paths.output_json.read_text(encoding="utf-8"))
        labels = payload["point_labels"]
        big = payload["is_big_gaussian"]
        transparent = payload["is_transparent_gaissian"]
        instances = payload["instances"]
        if (
            not isinstance(labels, list)
            or len(labels) != point_count
            or not isinstance(big, list)
            or len(big) != point_count
            or not isinstance(transparent, list)
            or len(transparent) != point_count
            or not isinstance(instances, Mapping)
        ):
            return False
        if any(not isinstance(value, int) for value in labels):
            return False
        for raw_id, metadata in instances.items():
            instance_id = int(raw_id)
            if instance_id not in labels or not isinstance(metadata, Mapping):
                return False
            if metadata.get("class") not in taxonomy.selected_classes:
                return False
            bbox = metadata.get("bbox")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 24
                or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in bbox
                )
            ):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def validate_source_workspace(workspace: SourceWorkspace) -> None:
    spec = SOURCE_VARIANTS[workspace.variant_id]
    required = (
        "grounded_SAM_masks.py",
        "get_scale.py",
        "train_contrastive_feature.py",
        "postprocess.py",
    )
    missing = [name for name in required if not (workspace.root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{workspace.variant_id} workspace is missing: {', '.join(missing)}"
        )
    text = (workspace.root / "train_contrastive_feature.py").read_text(encoding="utf-8")
    if (
        workspace.variant_id in {"full950", "full950-iterations-cli", "tip8c"}
        and not (workspace.root / "utils/resource_exit.py").is_file()
    ):
        raise FileNotFoundError(
            f"{workspace.variant_id} workspace is missing utils/resource_exit.py"
        )
    has_args = "def training(args, dataset," in text
    has_dim = "normalize(sample_features[None,...]*gates[:,None,...], dim=-1)" in text
    has_sorted = "masks = viewpoint_cam.original_masks.cuda()[sort_indices]" in text
    observed = (has_args, has_dim, has_sorted)
    expected = (
        spec.has_args_plumbing,
        spec.normalized_feature_dim,
        spec.sorted_semantic_masks,
    )
    if observed != expected:
        raise ValueError(
            f"{workspace.variant_id} source sentinels are {observed}, expected {expected}"
        )
    arguments = workspace.root / "arguments/__init__.py"
    if spec.integer_iteration_cli:
        argument_text = arguments.read_text(encoding="utf-8")
        if "self.iterations = 0" not in argument_text:
            raise ValueError(
                f"{workspace.variant_id} is missing the integer iterations CLI repair"
            )


def assert_isolated_output(
    output_root: Path,
    scenes: Sequence[RuntimeScene],
    workspaces: Sequence[SourceWorkspace],
) -> None:
    target = output_root.resolve()
    protected = [scene.base_path.resolve() for scene in scenes]
    protected.extend(workspace.root.resolve() for workspace in workspaces)
    for root in protected:
        if target == root or root in target.parents or target in root.parents:
            raise ValueError(
                f"output root must be disjoint from source/input assets: {target} vs {root}"
            )


def validate_scene_inputs(scene: RuntimeScene) -> None:
    required = (
        scene.images_path,
        scene.sparse_path,
        scene.point_cloud_path,
        scene.python_bin,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{scene.scene_id} input assets are missing: {', '.join(missing)}"
        )
    if read_ply_vertex_count(scene.point_cloud_path) in (None, 0):
        raise ValueError(f"invalid RGB Gaussian PLY: {scene.point_cloud_path}")
