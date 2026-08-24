from __future__ import annotations

import json
from pathlib import Path

import pytest

import category_priors.baseline_closure as closure
from category_priors.baseline_closure import (
    BFC18_CLASSES,
    BFC18_OTHER_CLASSES,
    CLOSURE_SCENES,
    REGISTERED_RUNS,
    TAXONOMIES,
    RuntimeScene,
    SourceWorkspace,
    assert_isolated_output,
    asset_paths,
    feature_is_complete,
    feature_paths,
    load_runtime_scenes,
    load_source_workspaces,
    masks_are_complete,
    output_is_complete,
    output_paths,
    record_masks_completion,
    scales_are_complete,
    validate_source_workspace,
)
from category_priors.baseline_closure_runner import (
    DISABLED_OTHER_CLASS,
    build_masks_invocation,
    build_postprocess_invocation,
    build_train_invocation,
    execute_stage,
    validate_cgroup,
)


class _Tensor:
    def __init__(self, shape) -> None:
        self.shape = shape


def _write_ply(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {count}\nproperty float x\nend_header\n"
        ).encode("ascii")
    )


def _scene(tmp_path: Path, point_count: int = 3) -> RuntimeScene:
    base = tmp_path / "scene"
    (base / "fastRecon/dense/sparse/0/images").mkdir(parents=True)
    (base / "fastRecon/dense/sparse/0/images/000.jpg").write_bytes(b"jpg")
    (base / "fastRecon/dense/sparse/0/cameras.bin").write_bytes(b"camera")
    _write_ply(
        base / "output_models/point_cloud/iteration_30000/point_cloud.ply",
        point_count,
    )
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    return RuntimeScene("scene0064_01", base, python)


def test_registered_closeout_is_bfc_primary_and_tip28_is_not_run() -> None:
    assert CLOSURE_SCENES == ("scene0064_01", "scene0025_01", "scene0231_00")
    assert len(BFC18_CLASSES) == 18
    assert BFC18_OTHER_CLASSES == ("socket", "book", "remote", "key")
    assert {run.variant_id for run in REGISTERED_RUNS} == {
        "literal-bfc",
        "args-only",
        "args-norm",
        "full950",
        "full950-iterations-cli",
    }
    assert all(run.variant_id != "tip8c" for run in REGISTERED_RUNS)
    literal = REGISTERED_RUNS[0]
    assert literal.scene_ids == ("scene0064_01",)
    assert literal.budget == "adaptive"
    assert literal.conditions == ("B0-global", "B1-original")
    full_adaptive = next(
        run
        for run in REGISTERED_RUNS
        if run.variant_id == "full950" and run.budget == "adaptive"
    )
    assert full_adaptive.scene_ids == CLOSURE_SCENES
    full_10k = next(run for run in REGISTERED_RUNS if run.budget == "10000")
    assert full_10k.variant_id == "full950-iterations-cli"
    assert full_10k.scene_ids == ("scene0064_01",)


def test_paths_keep_assets_features_and_outputs_disjoint(tmp_path: Path) -> None:
    assets = asset_paths(tmp_path, "scene0064_01")
    adaptive = feature_paths(tmp_path, "scene0064_01", "full950", "adaptive")
    control = feature_paths(tmp_path, "scene0064_01", "full950-iterations-cli", "10000")
    b0 = output_paths(tmp_path, "scene0064_01", "full950", "adaptive", "B0-global")
    b1 = output_paths(tmp_path, "scene0064_01", "full950", "adaptive", "B1-original")
    assert assets.root == tmp_path / "assets/bfc18/scene0064_01"
    assert adaptive.root != control.root
    assert b0.root != b1.root
    assert not (
        {adaptive.point_cloud, control.point_cloud} & {b0.output_json, b1.output_json}
    )


def test_runtime_and_workspace_manifests_are_explicit(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "scenes": {
                    "scene0064_01": {
                        "base_path": "/data/scene0064_01",
                        "python_bin": "/env/bin/python",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert load_runtime_scenes(runtime)["scene0064_01"].base_path == Path(
        "/data/scene0064_01"
    )
    workspaces = tmp_path / "workspaces.json"
    workspaces.write_text(
        json.dumps(
            {
                "workspaces": {
                    "literal-bfc": "/src/bfc",
                    "full950": {"root": "/src/full950"},
                }
            }
        ),
        encoding="utf-8",
    )
    loaded = load_source_workspaces(workspaces)
    assert loaded["literal-bfc"].root == Path("/src/bfc")
    assert loaded["full950"].root == Path("/src/full950")

    runtime.write_text(
        json.dumps(
            {
                "scenes": [
                    {"scene_id": "same", "base_path": "/a", "python_bin": "/p"},
                    {"scene_id": "same", "base_path": "/b", "python_bin": "/p"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate runtime scene"):
        load_runtime_scenes(runtime)


def test_output_root_must_not_overlap_input_or_source(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = SourceWorkspace("full950", workspace_root)
    separate = tmp_path.parent / f"{tmp_path.name}-outputs"
    assert_isolated_output(separate, (scene,), (workspace,))
    with pytest.raises(ValueError, match="disjoint"):
        assert_isolated_output(scene.base_path / "new-output", (scene,), (workspace,))


def test_teacher_scene_point_cloud_name_takes_priority(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    generic = scene.point_cloud_path
    teacher = generic.with_name("scene_point_cloud.ply")
    _write_ply(teacher, 3)
    assert scene.point_cloud_path == teacher


def test_commands_freeze_taxonomy_budget_and_b0_b1_difference(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    workspace = SourceWorkspace("full950", tmp_path / "full950")
    budget_workspace = SourceWorkspace(
        "full950-iterations-cli", tmp_path / "full950-iterations-cli"
    )
    taxonomy = TAXONOMIES["bfc18"]
    assets = asset_paths(tmp_path / "runs", scene.scene_id)
    adaptive = feature_paths(tmp_path / "runs", scene.scene_id, "full950", "adaptive")
    control = feature_paths(
        tmp_path / "runs",
        scene.scene_id,
        "full950-iterations-cli",
        "10000",
    )
    masks = build_masks_invocation(
        scene,
        assets,
        workspace,
        taxonomy,
        sam_checkpoint=tmp_path / "sam.pth",
        groundingdino_checkpoint=tmp_path / "dino.pth",
        groundingdino_config=tmp_path / "dino.py",
    )
    class_index = masks.command.index("--classes")
    assert masks.command[class_index + 1 :] == BFC18_CLASSES

    adaptive_train = build_train_invocation(
        scene, assets, adaptive, workspace, budget="adaptive"
    )
    control_train = build_train_invocation(
        scene, assets, control, budget_workspace, budget="10000"
    )
    assert "--iterations" not in adaptive_train.command
    iteration_index = control_train.command.index("--iterations")
    assert control_train.command[iteration_index + 1] == "10000"
    with pytest.raises(ValueError, match="integer iterations CLI"):
        build_train_invocation(scene, assets, control, workspace, budget="10000")

    b0_paths = output_paths(
        tmp_path / "runs", scene.scene_id, "full950", "adaptive", "B0-global"
    )
    b1_paths = output_paths(
        tmp_path / "runs", scene.scene_id, "full950", "adaptive", "B1-original"
    )
    b0 = build_postprocess_invocation(
        scene,
        assets,
        adaptive,
        b0_paths,
        workspace,
        taxonomy,
        budget="adaptive",
        condition="B0-global",
    )
    b1 = build_postprocess_invocation(
        scene,
        assets,
        adaptive,
        b1_paths,
        workspace,
        taxonomy,
        budget="adaptive",
        condition="B1-original",
    )
    b0_index = b0.command.index("--other_classes")
    b1_index = b1.command.index("--other_classes")
    assert b0.command[b0_index + 1 :] == (DISABLED_OTHER_CLASS,)
    assert b1.command[b1_index + 1 :] == BFC18_OTHER_CLASSES
    forbidden = {"--clean", "download", "sha256", "lock", "cache"}
    assert not forbidden.intersection(item.lower() for item in b0.command + b1.command)


@pytest.mark.parametrize(
    ("variant", "has_args", "has_dim", "has_sorted"),
    (
        ("literal-bfc", False, False, False),
        ("args-only", True, False, False),
        ("args-norm", True, True, False),
        ("full950", True, True, True),
        ("full950-iterations-cli", True, True, True),
    ),
)
def test_source_variant_sentinels(
    tmp_path: Path, variant: str, has_args: bool, has_dim: bool, has_sorted: bool
) -> None:
    root = tmp_path / variant
    root.mkdir()
    if variant in {"full950", "full950-iterations-cli"}:
        (root / "utils").mkdir()
        (root / "utils/resource_exit.py").write_text("# present\n", encoding="utf-8")
    if variant == "full950-iterations-cli":
        (root / "arguments").mkdir()
        (root / "arguments/__init__.py").write_text(
            "self.iterations = 0\n", encoding="utf-8"
        )
    for name in ("grounded_SAM_masks.py", "get_scale.py", "postprocess.py"):
        (root / name).write_text("# present\n", encoding="utf-8")
    lines = []
    if has_args:
        lines.append("def training(args, dataset, opt, pipe, iteration): pass")
    else:
        lines.append("def training(dataset, opt, pipe, iteration): pass")
    if has_dim:
        lines.append("normalize(sample_features[None,...]*gates[:,None,...], dim=-1)")
    else:
        lines.append("normalize(sample_features[None,...]*gates[:,None,...])")
    if has_sorted:
        lines.append("masks = viewpoint_cam.original_masks.cuda()[sort_indices]")
    else:
        lines.append("masks = viewpoint_cam.original_masks.cuda()")
    (root / "train_contrastive_feature.py").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    validate_source_workspace(SourceWorkspace(variant, root))
    with (root / "train_contrastive_feature.py").open("a", encoding="utf-8") as handle:
        handle.write("\nmasks = viewpoint_cam.original_masks.cuda()[sort_indices]\n")
    if not has_sorted:
        with pytest.raises(ValueError, match="source sentinels"):
            validate_source_workspace(SourceWorkspace(variant, root))


def test_stage_completeness_rejects_corrupt_or_mismatched_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _scene(tmp_path)
    paths = asset_paths(tmp_path / "runs", scene.scene_id)
    paths.masks.mkdir(parents=True)
    paths.labels.mkdir(parents=True)
    paths.mask_scales.mkdir(parents=True)
    paths.masks_progress.parent.mkdir(parents=True)
    paths.masks_progress.write_text("100", encoding="utf-8")
    paths.scale_progress.write_text("100", encoding="utf-8")
    for path in (
        paths.masks / "000.pt",
        paths.labels / "000.pt",
        paths.label_features,
        paths.mask_scales / "000.pt",
    ):
        path.write_bytes(b"tensor")

    def fake_load(path: Path):
        if path == paths.label_features:
            return _Tensor((18, 32))
        if path.parent == paths.masks:
            return _Tensor((2, 8, 8))
        if path.parent == paths.labels:
            return _Tensor((2,))
        if path.parent == paths.mask_scales:
            return _Tensor((2,))
        raise AssertionError(path)

    monkeypatch.setattr(closure, "_load_torch", fake_load)
    record_masks_completion(scene, paths, TAXONOMIES["bfc18"])
    assert masks_are_complete(scene, paths, TAXONOMIES["bfc18"])
    assert scales_are_complete(paths)
    (paths.labels / "stray.pt").write_bytes(b"bad")
    assert not masks_are_complete(scene, paths, TAXONOMIES["bfc18"])
    (paths.labels / "stray.pt").unlink()
    (paths.mask_scales / "000.pt").unlink()
    assert not scales_are_complete(paths)


def test_feature_and_output_completeness_validate_point_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _scene(tmp_path, point_count=3)
    features = feature_paths(tmp_path / "runs", scene.scene_id, "full950", "adaptive")
    features.root.mkdir(parents=True)
    features.progress.write_text("100", encoding="utf-8")
    _write_ply(features.point_cloud, 3)
    features.scale_gate.write_bytes(b"state")
    monkeypatch.setattr(
        closure,
        "_load_torch",
        lambda _path: {"0.weight": _Tensor((32, 1)), "0.bias": _Tensor((32,))},
    )
    assert feature_is_complete(scene, features)
    _write_ply(features.point_cloud, 2)
    assert not feature_is_complete(scene, features)

    output = output_paths(
        tmp_path / "runs", scene.scene_id, "full950", "adaptive", "B1-original"
    )
    output.root.mkdir(parents=True)
    output.output_json.write_text(
        json.dumps(
            {
                "point_labels": [0, 0, -1],
                "is_big_gaussian": [False, False, False],
                "is_transparent_gaissian": [False, False, False],
                "instances": {"0": {"class": "chair", "bbox": [0.0] * 24}},
            }
        ),
        encoding="utf-8",
    )
    assert output_is_complete(scene, output, TAXONOMIES["bfc18"])
    payload = json.loads(output.output_json.read_text(encoding="utf-8"))
    payload["point_labels"] = [0, 7, -1]
    output.output_json.write_text(json.dumps(payload), encoding="utf-8")
    assert output_is_complete(scene, output, TAXONOMIES["bfc18"])
    payload["point_labels"] = [0]
    output.output_json.write_text(json.dumps(payload), encoding="utf-8")
    assert not output_is_complete(scene, output, TAXONOMIES["bfc18"])


def test_execute_stage_skips_complete_and_reruns_damaged(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    workspace = SourceWorkspace("full950", tmp_path / "full950")
    assets = asset_paths(tmp_path / "runs", scene.scene_id)
    features = feature_paths(tmp_path / "runs", scene.scene_id, "full950", "adaptive")
    invocation = build_train_invocation(
        scene, assets, features, workspace, budget="adaptive"
    )
    state = {"complete": True, "calls": 0}

    def executor(_invocation) -> int:
        state["calls"] += 1
        state["complete"] = True
        return 0

    reused = execute_stage(
        invocation, is_complete=lambda: state["complete"], executor=executor
    )
    assert reused["status"] == "reused"
    assert state["calls"] == 0
    state["complete"] = False
    completed = execute_stage(
        invocation, is_complete=lambda: state["complete"], executor=executor
    )
    assert completed["status"] == "completed"
    assert state["calls"] == 1


def test_cgroup_guard_reads_only_90_gib_contract(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.max").write_text(str(90 * 1024**3), encoding="utf-8")
    (cgroup / "memory.current").write_text(str(12 * 1024**3), encoding="utf-8")
    (cgroup / "memory.events").write_text(
        "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n", encoding="utf-8"
    )
    snapshot = validate_cgroup(cgroup)
    assert snapshot is not None
    assert snapshot["max"] == 90 * 1024**3
    assert snapshot["events"]["oom_kill"] == 0
    (cgroup / "memory.max").write_text("max", encoding="utf-8")
    with pytest.raises(RuntimeError, match="90 GiB"):
        validate_cgroup(cgroup)
