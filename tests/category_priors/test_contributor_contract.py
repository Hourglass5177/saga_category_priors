from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "submodules/diff-gaussian-rasterization-max-contributor"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_max_contributor_uses_preupdate_transmittance() -> None:
    source = _read(
        "submodules/diff-gaussian-rasterization-max-contributor/"
        "cuda_rasterizer/forward.cu"
    )
    weight = source.index("const float weight = alpha * T;")
    compare = source.index("if(weight > max_weight)", weight)
    update = source.index("T = test_T;", weight)
    assert weight < compare < update


def test_empty_pixels_use_invalid_contributor_id_and_zero_weight() -> None:
    binding = _read(
        "submodules/diff-gaussian-rasterization-max-contributor/"
        "rasterize_points.cu"
    )
    forward = _read(
        "submodules/diff-gaussian-rasterization-max-contributor/"
        "cuda_rasterizer/forward.cu"
    )
    assert "torch::full({H, W}, -1" in binding
    assert "torch::full({H, W}, 0.0" in binding
    assert "int max_id = -1;" in forward
    assert "float max_weight = 0.0f;" in forward


def test_active_contributor_abi_has_no_historical_branch() -> None:
    paths = (
        ROOT / "gaussian_renderer/__init__.py",
        EXTENSION / "rasterize_points.h",
        EXTENSION / "rasterize_points.cu",
        EXTENSION / "cuda_rasterizer/forward.h",
        EXTENSION / "cuda_rasterizer/forward.cu",
        EXTENSION / "cuda_rasterizer/rasterizer.h",
        EXTENSION / "cuda_rasterizer/rasterizer_impl.h",
        EXTENSION / "cuda_rasterizer/rasterizer_impl.cu",
        EXTENSION / "diff_gaussian_rasterization_max_contributor/__init__.py",
    )
    for path in paths:
        assert "historical_max" not in path.read_text(encoding="utf-8")


def test_python_rasterizer_returns_only_color_contributor_weight_and_radii() -> None:
    source = _read(
        "submodules/diff-gaussian-rasterization-max-contributor/"
        "diff_gaussian_rasterization_max_contributor/__init__.py"
    )
    assert "return color, max_contributor, max_contribute, radii" in source
    assert (
        "def backward(ctx, grad_out_color, _max_id, _max_weight, _radii):"
        in source
    )


def test_public_renderer_documents_corrected_contract() -> None:
    source = _read("gaussian_renderer/__init__.py")
    assert "alpha * T_prev" in source
    assert "explicit sentinel ``-1 / 0``" in source
    assert "rendered_image, max_contributor, max_contribute, radii = rasterizer(" in source
