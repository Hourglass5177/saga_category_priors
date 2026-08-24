from __future__ import annotations

from pathlib import Path

import pytest

from category_priors.baseline_closure_contributor import (
    RASTERIZER_RELATIVE_PATH,
    ContributorVariantError,
    materialize_contributor_fixed_variant,
)

HISTORICAL_RENDER_UNPACK = (
    "    rendered_image, max_contributor, max_contribute, radii = rasterizer("
)
HISTORICAL_RENDER_ID = "            'max_contributor': max_contributor,"
HISTORICAL_RENDER_WEIGHT = "            'max_contribute': max_contribute,"
HISTORICAL_POSTPROCESS_LOAD = """        max_contributor = render_pkg['max_contributor'].to(point_labels.device)
        max_contribute = render_pkg['max_contribute'].to(point_labels.device)
        max_instance_contributor = point_labels[max_contributor]"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_full950_tree(root: Path) -> Path:
    _write(
        root / "gaussian_renderer" / "__init__.py",
        "\n".join(  # noqa: FLY002 - source fixture is assembled from sentinels
            (
                "def render_with_max_contributor():",
                HISTORICAL_RENDER_UNPACK,
                "        means3D=means3D)",
                "    return {",
                HISTORICAL_RENDER_ID,
                HISTORICAL_RENDER_WEIGHT,
                "    }",
                "",
            )
        ),
    )
    _write(
        root / "postprocess.py",
        "\n".join(  # noqa: FLY002 - source fixture is assembled from sentinels
            (
                "def vote():",
                HISTORICAL_POSTPROCESS_LOAD,
                "        for label, mask in zip(labels, masks):",
                "            vote_for_label = max_instance_contributor[mask]",
                "        vote_for_background_label = max_instance_contributor[background]",
                "",
            )
        ),
    )
    _write(root / RASTERIZER_RELATIVE_PATH / "stale.txt", "historical\n")
    _write(root / "copied.txt", "keep me\n")
    _write(root / ".git" / "config", "do not copy\n")
    return root


def _make_dual_rasterizer_tree(root: Path) -> Path:
    _write(
        root / "rasterize_points.cu",
        "\n".join(  # noqa: FLY002 - C++ fixture keeps literal braces
            (
                "torch::Tensor max_contributor = torch::full({H, W}, -1, options);",
                "return std::make_tuple(rendered, out_color, max_contributor, max_contribute, historical_max_contributor, historical_max_contribute, radii, geomBuffer, binningBuffer, imgBuffer);",
                "",
            )
        ),
    )
    _write(
        root / "cuda_rasterizer" / "forward.cu",
        "\n".join(  # noqa: FLY002 - source fixture is intentionally line based
            (
                "int max_id = -1;",
                "const float weight = alpha * T;",
                "const float historical_weight = alpha * test_T;",
                "",
            )
        ),
    )
    _write(
        root / "diff_gaussian_rasterization_max_contributor" / "__init__.py",
        "\n".join(  # noqa: FLY002 - source fixture is intentionally line based
            (
                "return color, max_contributor, max_contribute, historical_max_contributor, historical_max_contribute, radii",
                "def backward(ctx, grad_out_color, _fixed_id, _fixed_weight, _historical_id, _historical_weight, _radii):",
                "    pass",
                "",
            )
        ),
    )
    _write(root / "dual.txt", "fixed and historical outputs\n")
    _write(root / "build" / "ignored.bin", "compiled\n")
    return root


def test_materializes_fixed_variant_without_mutating_inputs(tmp_path: Path) -> None:
    full950 = _make_full950_tree(tmp_path / "full950")
    rasterizer = _make_dual_rasterizer_tree(tmp_path / "dual-rasterizer")
    output = tmp_path / "materialized"

    result = materialize_contributor_fixed_variant(full950, rasterizer, output)

    assert result["contributor_mode"] == "fixed-alpha-times-t-prev"
    assert result["output_root"] == str(output.resolve())
    assert (output / "copied.txt").read_text(encoding="utf-8") == "keep me\n"
    assert not (output / ".git").exists()
    assert not (output / RASTERIZER_RELATIVE_PATH / "stale.txt").exists()
    assert (output / RASTERIZER_RELATIVE_PATH / "dual.txt").is_file()
    assert not (output / RASTERIZER_RELATIVE_PATH / "build").exists()

    renderer = (output / "gaussian_renderer" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "fixed_max_contributor, fixed_max_contribute" in renderer
    assert "_historical_max_contributor, _historical_max_contribute" in renderer
    assert "'max_contributor': fixed_max_contributor" in renderer
    assert "'max_contribute': fixed_max_contribute" in renderer
    assert HISTORICAL_RENDER_UNPACK not in renderer

    postprocess = (output / "postprocess.py").read_text(encoding="utf-8")
    assert "(max_contributor >= 0)" in postprocess
    assert "(max_contribute > 0)" in postprocess
    assert "max_contributor < point_labels.shape[0]" in postprocess
    assert "mask & valid_contributor" in postprocess
    assert "background & valid_contributor" in postprocess
    assert HISTORICAL_POSTPROCESS_LOAD not in postprocess

    original_renderer = (full950 / "gaussian_renderer" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert HISTORICAL_RENDER_UNPACK in original_renderer
    assert (full950 / RASTERIZER_RELATIVE_PATH / "stale.txt").is_file()


def test_refuses_existing_output_tree(tmp_path: Path) -> None:
    full950 = _make_full950_tree(tmp_path / "full950")
    rasterizer = _make_dual_rasterizer_tree(tmp_path / "dual-rasterizer")
    output = tmp_path / "materialized"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        materialize_contributor_fixed_variant(full950, rasterizer, output)

    assert list(output.iterdir()) == []


def test_rejects_ambiguous_historical_sentinel_without_partial_output(
    tmp_path: Path,
) -> None:
    full950 = _make_full950_tree(tmp_path / "full950")
    renderer_path = full950 / "gaussian_renderer" / "__init__.py"
    renderer_path.write_text(
        renderer_path.read_text(encoding="utf-8") + HISTORICAL_RENDER_UNPACK + "\n",
        encoding="utf-8",
    )
    rasterizer = _make_dual_rasterizer_tree(tmp_path / "dual-rasterizer")
    output = tmp_path / "materialized"

    with pytest.raises(ContributorVariantError, match="exactly once; found 2"):
        materialize_contributor_fixed_variant(full950, rasterizer, output)

    assert not output.exists()


def test_rejects_non_dual_rasterizer_without_partial_output(tmp_path: Path) -> None:
    full950 = _make_full950_tree(tmp_path / "full950")
    rasterizer = _make_dual_rasterizer_tree(tmp_path / "dual-rasterizer")
    forward = rasterizer / "cuda_rasterizer" / "forward.cu"
    forward.write_text(
        forward.read_text(encoding="utf-8").replace(
            "const float historical_weight = alpha * test_T;", ""
        ),
        encoding="utf-8",
    )
    output = tmp_path / "materialized"

    with pytest.raises(ContributorVariantError, match="found 0"):
        materialize_contributor_fixed_variant(full950, rasterizer, output)

    assert not output.exists()


def test_rejects_output_nested_inside_an_input_tree(tmp_path: Path) -> None:
    full950 = _make_full950_tree(tmp_path / "full950")
    rasterizer = _make_dual_rasterizer_tree(tmp_path / "dual-rasterizer")

    with pytest.raises(ContributorVariantError, match="must not be inside"):
        materialize_contributor_fixed_variant(
            full950, rasterizer, full950 / "generated"
        )
