from pathlib import Path


def test_corrected_max_contributor_uses_preupdate_transmittance() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "submodules/diff-gaussian-rasterization-max-contributor/cuda_rasterizer/forward.cu"
    ).read_text(encoding="utf-8")
    weight = source.index("const float weight = alpha * T;")
    compare = source.index("if(weight > max_weight)", weight)
    update = source.index("T = test_T;", weight)
    assert weight < compare < update


def test_empty_pixels_use_invalid_contributor_id() -> None:
    root = Path(__file__).resolve().parents[2]
    forward = (root / "submodules/diff-gaussian-rasterization-max-contributor/cuda_rasterizer/forward.cu").read_text(encoding="utf-8")
    binding = (root / "submodules/diff-gaussian-rasterization-max-contributor/rasterize_points.cu").read_text(encoding="utf-8")
    assert "int max_id = -1;" in forward
    assert "torch::full({H, W}, -1" in binding
