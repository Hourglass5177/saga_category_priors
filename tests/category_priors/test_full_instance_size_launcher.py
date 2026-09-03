from __future__ import annotations

from pathlib import Path


def _launcher() -> str:
    return (
        Path(__file__).resolve().parents[2]
        / "continue_full_instance_size_prior.sh"
    ).read_text(encoding="utf-8")


def test_launcher_binds_registered_rasterizers_and_checks_clean_tree() -> None:
    source = _launcher()
    assert "git diff --quiet --" in source
    assert "git diff --cached --quiet --" in source
    assert "$workspace/submodules/diff-gaussian-rasterization:" in source
    assert (
        "$workspace/submodules/diff-gaussian-rasterization-max-contributor:"
        in source
    )
    assert "import pandas" in source
    assert "import pyarrow" in source
    assert "torch.cuda.is_available()" in source
    assert "from diff_gaussian_rasterization import _C" in source
    assert "from diff_gaussian_rasterization_max_contributor import (" in source
    assert "from gaussian_renderer import" in source


def test_launcher_executes_discriminating_contributor_smoke() -> None:
    source = _launcher()
    heredoc = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    compile(heredoc, "continue_full_instance_size_prior.sh:preflight", "exec")
    assert "[[0.30], [0.50]]" in heredoc
    assert "fixed_ids[centre].item()) != 1" in heredoc
    assert "historical_ids[centre].item()) != 0" in heredoc
    assert "fixed_ids[0, 0].item()) != -1" in heredoc
    assert "fixed_weights[0, 0].item()) != 0.0" in heredoc


def test_launcher_passes_all_split_inputs_and_rebuilds_missing_t1() -> None:
    source = _launcher()
    for option in (
        "--runtime-manifest",
        "--locked-runtime-manifest",
        "--gt-dir",
        "--locked-gt-dir",
        "--train-stats",
        "--category-priors",
        "--size-bins",
        "--locked-evaluation-scenes",
        "--python-bin",
    ):
        assert option in source
    assert "--allow-rebuild-missing-traces" in source
    assert '--python-bin "$python_bin"' in source
    assert "export CUDA_VISIBLE_DEVICES=0" in source
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"' not in source
