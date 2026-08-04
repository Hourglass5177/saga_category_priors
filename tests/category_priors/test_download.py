from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from category_priors.download import (
    MINIMAL_FILE_TYPES,
    _safe_error,
    build_download_tasks,
    download_scannet_subset,
)


def _official_module() -> SimpleNamespace:
    return SimpleNamespace(
        BASE_URL="https://restricted.invalid/",
        RELEASE="v2/scans",
        RELEASE_TASKS="v2/tasks",
        LABEL_MAP_FILE="scannetv2-labels.combined.tsv",
    )


def test_build_tasks_is_minimal_and_does_not_include_sens(tmp_path: Path) -> None:
    tasks = build_download_tasks(
        _official_module(), ["scene0000_00"], tmp_path, include_label_map=True
    )
    assert len(tasks) == len(MINIMAL_FILE_TYPES) + 1
    assert all(".sens" not in task.suffix for task in tasks)
    assert tasks[0].target.parent == tmp_path / "scans" / "scene0000_00"


def test_build_tasks_rejects_unregistered_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="registered statistics"):
        build_download_tasks(
            _official_module(),
            ["scene0000_00"],
            tmp_path,
            file_types=(".sens",),
        )


def test_download_requires_explicit_tos_acceptance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Terms of Use"):
        download_scannet_subset(
            official_downloader=tmp_path / "download.py",
            scene_list=tmp_path / "split.txt",
            out_dir=tmp_path / "data",
            manifest_path=tmp_path / "manifest.json",
            accept_tos=False,
        )


def test_safe_error_does_not_echo_private_url() -> None:
    error = ValueError("https://restricted.invalid/private-token")
    assert _safe_error(error) == "ValueError"
    assert "restricted" not in _safe_error(error)
