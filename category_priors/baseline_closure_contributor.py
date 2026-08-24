from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

RASTERIZER_RELATIVE_PATH = Path(
    "submodules/diff-gaussian-rasterization-max-contributor"
)


class ContributorVariantError(RuntimeError):
    """Raised when a historical/fixed source sentinel is not exact."""


_HISTORICAL_RENDER_UNPACK = (
    "    rendered_image, max_contributor, max_contribute, radii = rasterizer("
)
_FIXED_RENDER_UNPACK = (
    "    rendered_image, fixed_max_contributor, fixed_max_contribute, "
    "_historical_max_contributor, _historical_max_contribute, radii = rasterizer("
)
_HISTORICAL_RENDER_ID = "            'max_contributor': max_contributor,"
_FIXED_RENDER_ID = "            'max_contributor': fixed_max_contributor,"
_HISTORICAL_RENDER_WEIGHT = "            'max_contribute': max_contribute,"
_FIXED_RENDER_WEIGHT = "            'max_contribute': fixed_max_contribute,"

_HISTORICAL_POSTPROCESS_LOAD = """        max_contributor = render_pkg['max_contributor'].to(point_labels.device)
        max_contribute = render_pkg['max_contribute'].to(point_labels.device)
        max_instance_contributor = point_labels[max_contributor]"""
_FIXED_POSTPROCESS_LOAD = """        max_contributor = render_pkg['max_contributor'].to(point_labels.device)
        max_contribute = render_pkg['max_contribute'].to(point_labels.device)
        valid_contributor = (
            (max_contributor >= 0)
            & (max_contributor < point_labels.shape[0])
            & (max_contribute > 0)
        )
        safe_contributor = max_contributor.clone()
        safe_contributor[~valid_contributor] = 0
        max_instance_contributor = point_labels[safe_contributor]"""
_HISTORICAL_FOREGROUND_VOTE = (
    "            vote_for_label = max_instance_contributor[mask]"
)
_FIXED_FOREGROUND_VOTE = (
    "            vote_for_label = max_instance_contributor[mask & valid_contributor]"
)
_HISTORICAL_BACKGROUND_VOTE = (
    "        vote_for_background_label = max_instance_contributor[background]"
)
_FIXED_BACKGROUND_VOTE = (
    "        vote_for_background_label = "
    "max_instance_contributor[background & valid_contributor]"
)

_DUAL_RASTERIZER_SENTINELS: dict[Path, tuple[str, ...]] = {
    Path("rasterize_points.cu"): (
        "torch::Tensor max_contributor = torch::full({H, W}, -1,",
        (
            "return std::make_tuple(rendered, out_color, max_contributor, "
            "max_contribute, historical_max_contributor, "
            "historical_max_contribute, radii, geomBuffer, binningBuffer, "
            "imgBuffer);"
        ),
    ),
    Path("cuda_rasterizer/forward.cu"): (
        "int max_id = -1;",
        "const float weight = alpha * T;",
        "const float historical_weight = alpha * test_T;",
    ),
    Path("diff_gaussian_rasterization_max_contributor/__init__.py"): (
        (
            "return color, max_contributor, max_contribute, "
            "historical_max_contributor, historical_max_contribute, radii"
        ),
        (
            "def backward(ctx, grad_out_color, _fixed_id, _fixed_weight, "
            "_historical_id, _historical_weight, _radii):"
        ),
    ),
}

_COPY_IGNORE = shutil.ignore_patterns(
    ".git",
    "__pycache__",
    "*.pyc",
    "build",
    "dist",
    "*.egg-info",
)


def _read_text(path: Path, label: str) -> str:
    if not path.is_file():
        raise ContributorVariantError(f"{label} file is missing: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ContributorVariantError(f"{label} is not UTF-8: {path}") from exc


def _require_once(text: str, sentinel: str, label: str) -> None:
    count = text.count(sentinel)
    if count != 1:
        raise ContributorVariantError(
            f"{label} sentinel must occur exactly once; found {count}: {sentinel!r}"
        )


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    _require_once(text, old, label)
    if new in text:
        raise ContributorVariantError(
            f"{label} already contains the fixed sentinel: {new!r}"
        )
    return text.replace(old, new, 1)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_inputs(
    full950_root: Path,
    fixed_rasterizer_root: Path,
    output_root: Path,
) -> tuple[str, str]:
    if not full950_root.is_dir():
        raise ContributorVariantError(f"full950 source tree is missing: {full950_root}")
    if not fixed_rasterizer_root.is_dir():
        raise ContributorVariantError(
            f"fixed rasterizer source tree is missing: {fixed_rasterizer_root}"
        )
    if output_root.exists():
        raise FileExistsError(f"output tree already exists: {output_root}")

    resolved_full = full950_root.resolve()
    resolved_rasterizer = fixed_rasterizer_root.resolve()
    resolved_output = output_root.resolve()
    if _is_within(resolved_output, resolved_full):
        raise ContributorVariantError(
            "output tree must not be inside the full950 source tree"
        )
    if _is_within(resolved_output, resolved_rasterizer):
        raise ContributorVariantError(
            "output tree must not be inside the fixed rasterizer source tree"
        )

    historical_rasterizer = full950_root / RASTERIZER_RELATIVE_PATH
    if not historical_rasterizer.is_dir():
        raise ContributorVariantError(
            f"full950 rasterizer directory is missing: {historical_rasterizer}"
        )

    renderer_path = full950_root / "gaussian_renderer" / "__init__.py"
    postprocess_path = full950_root / "postprocess.py"
    renderer_text = _read_text(renderer_path, "full950 gaussian renderer")
    postprocess_text = _read_text(postprocess_path, "full950 postprocess")

    for sentinel, label in (
        (_HISTORICAL_RENDER_UNPACK, "full950 renderer unpack"),
        (_HISTORICAL_RENDER_ID, "full950 renderer contributor ID"),
        (_HISTORICAL_RENDER_WEIGHT, "full950 renderer contributor weight"),
    ):
        _require_once(renderer_text, sentinel, label)
    for sentinel, label in (
        (_HISTORICAL_POSTPROCESS_LOAD, "full950 postprocess contributor load"),
        (_HISTORICAL_FOREGROUND_VOTE, "full950 postprocess foreground vote"),
        (_HISTORICAL_BACKGROUND_VOTE, "full950 postprocess background vote"),
    ):
        _require_once(postprocess_text, sentinel, label)

    for relative_path, sentinels in _DUAL_RASTERIZER_SENTINELS.items():
        source_text = _read_text(
            fixed_rasterizer_root / relative_path,
            f"fixed rasterizer {relative_path.as_posix()}",
        )
        for sentinel in sentinels:
            _require_once(
                source_text,
                sentinel,
                f"fixed rasterizer {relative_path.as_posix()}",
            )
    return renderer_text, postprocess_text


def _patch_renderer(text: str) -> str:
    patched = _replace_once(
        text,
        _HISTORICAL_RENDER_UNPACK,
        _FIXED_RENDER_UNPACK,
        "full950 renderer unpack",
    )
    patched = _replace_once(
        patched,
        _HISTORICAL_RENDER_ID,
        _FIXED_RENDER_ID,
        "full950 renderer contributor ID",
    )
    return _replace_once(
        patched,
        _HISTORICAL_RENDER_WEIGHT,
        _FIXED_RENDER_WEIGHT,
        "full950 renderer contributor weight",
    )


def _patch_postprocess(text: str) -> str:
    patched = _replace_once(
        text,
        _HISTORICAL_POSTPROCESS_LOAD,
        _FIXED_POSTPROCESS_LOAD,
        "full950 postprocess contributor load",
    )
    patched = _replace_once(
        patched,
        _HISTORICAL_FOREGROUND_VOTE,
        _FIXED_FOREGROUND_VOTE,
        "full950 postprocess foreground vote",
    )
    return _replace_once(
        patched,
        _HISTORICAL_BACKGROUND_VOTE,
        _FIXED_BACKGROUND_VOTE,
        "full950 postprocess background vote",
    )


def materialize_contributor_fixed_variant(
    full950_root: str | Path,
    fixed_rasterizer_root: str | Path,
    output_root: str | Path,
) -> dict[str, str]:
    """Create a fixed-contributor source tree from one exact full950 tree.

    The operation never mutates either input.  Historical source sentinels and
    the dual-output rasterizer ABI must match exactly before any copy starts.
    The destination must not already exist.
    """

    full950 = Path(full950_root)
    fixed_rasterizer = Path(fixed_rasterizer_root)
    output = Path(output_root)
    renderer_text, postprocess_text = _validate_inputs(
        full950, fixed_rasterizer, output
    )
    patched_renderer = _patch_renderer(renderer_text)
    patched_postprocess = _patch_postprocess(postprocess_text)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(
            full950,
            temporary,
            symlinks=True,
            ignore=_COPY_IGNORE,
        )
        destination_rasterizer = temporary / RASTERIZER_RELATIVE_PATH
        shutil.rmtree(destination_rasterizer)
        shutil.copytree(
            fixed_rasterizer,
            destination_rasterizer,
            symlinks=True,
            ignore=_COPY_IGNORE,
        )
        (temporary / "gaussian_renderer" / "__init__.py").write_text(
            patched_renderer, encoding="utf-8"
        )
        (temporary / "postprocess.py").write_text(patched_postprocess, encoding="utf-8")

        copied_renderer = _read_text(
            temporary / "gaussian_renderer" / "__init__.py",
            "materialized gaussian renderer",
        )
        copied_postprocess = _read_text(
            temporary / "postprocess.py", "materialized postprocess"
        )
        for sentinel, label in (
            (_FIXED_RENDER_UNPACK, "materialized renderer fixed unpack"),
            (_FIXED_RENDER_ID, "materialized renderer fixed ID"),
            (_FIXED_RENDER_WEIGHT, "materialized renderer fixed weight"),
        ):
            _require_once(copied_renderer, sentinel, label)
        for sentinel, label in (
            (_FIXED_POSTPROCESS_LOAD, "materialized postprocess safe load"),
            (_FIXED_FOREGROUND_VOTE, "materialized foreground filter"),
            (_FIXED_BACKGROUND_VOTE, "materialized background filter"),
        ):
            _require_once(copied_postprocess, sentinel, label)
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "full950_root": str(full950.resolve()),
        "fixed_rasterizer_root": str(fixed_rasterizer.resolve()),
        "output_root": str(output.resolve()),
        "contributor_mode": "fixed-alpha-times-t-prev",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the contributor-fixed full950 source variant."
    )
    parser.add_argument("--full950-root", type=Path, required=True)
    parser.add_argument("--fixed-rasterizer-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_contributor_fixed_variant(
        args.full950_root,
        args.fixed_rasterizer_root,
        args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
