from __future__ import annotations

import importlib.util
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from .io import hash_json, sha256_file, write_json
from .scannet import read_scene_ids, validate_scene_ids

MINIMAL_FILE_TYPES = (
    ".aggregation.json",
    ".txt",
    "_vh_clean_2.ply",
    "_vh_clean_2.0.010000.segs.json",
)
_SCENE_ID = re.compile(r"^scene\d{4}_\d{2}$")


@dataclass(frozen=True)
class DownloadTask:
    scene_id: str | None
    suffix: str
    url: str
    target: Path


def load_official_downloader(path: str | Path) -> ModuleType:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Official ScanNet downloader not found: {source}")
    spec = importlib.util.spec_from_file_location("scannet_official_download", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official ScanNet downloader: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("BASE_URL", "RELEASE", "RELEASE_TASKS", "LABEL_MAP_FILE")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "Official ScanNet downloader is missing constants: " + ", ".join(missing)
        )
    return module


def build_download_tasks(
    module: ModuleType,
    scene_ids: list[str],
    out_dir: str | Path,
    file_types: tuple[str, ...] = MINIMAL_FILE_TYPES,
    include_label_map: bool = True,
) -> list[DownloadTask]:
    invalid_types = sorted(set(file_types) - set(MINIMAL_FILE_TYPES))
    if invalid_types:
        raise ValueError(
            "Only the registered statistics file types are allowed: "
            + ", ".join(invalid_types)
        )
    root = Path(out_dir).resolve()
    tasks: list[DownloadTask] = []
    base_url = str(module.BASE_URL)
    release = str(module.RELEASE).strip("/")
    for scene_id in validate_scene_ids(scene_ids):
        if not _SCENE_ID.fullmatch(scene_id):
            raise ValueError(f"Invalid ScanNet scene id: {scene_id}")
        for suffix in file_types:
            filename = f"{scene_id}{suffix}"
            tasks.append(
                DownloadTask(
                    scene_id=scene_id,
                    suffix=suffix,
                    url=f"{base_url}{release}/{scene_id}/{filename}",
                    target=root / "scans" / scene_id / filename,
                )
            )
    if include_label_map:
        label_map = str(module.LABEL_MAP_FILE)
        release_tasks = str(module.RELEASE_TASKS).strip("/")
        tasks.append(
            DownloadTask(
                scene_id=None,
                suffix=label_map,
                url=f"{base_url}{release_tasks}/{label_map}",
                target=root / label_map,
            )
        )
    return tasks


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTPError status={exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return f"URLError reason_type={type(exc.reason).__name__}"
    return type(exc).__name__


def _check_free_space(path: Path, min_free_bytes: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < min_free_bytes:
        raise RuntimeError(
            f"Free-space gate failed: {free / (1024**3):.1f}GB available, "
            f"{min_free_bytes / (1024**3):.1f}GB required"
        )


def _download_one(
    task: DownloadTask,
    *,
    root: Path,
    retries: int,
    timeout_s: float,
    min_free_bytes: int,
) -> dict[str, Any]:
    target = task.target
    relative = target.relative_to(root).as_posix()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return {
            "scene_id": task.scene_id,
            "suffix": task.suffix,
            "path": relative,
            "status": "existing",
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }

    partial = target.with_name(target.name + ".part")
    last_error = "unknown"
    for attempt in range(1, retries + 1):
        try:
            _check_free_space(root, min_free_bytes)
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(task.url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                status = getattr(response, "status", None)
                append = bool(offset and status == 206)
                mode = "ab" if append else "wb"
                starting_size = offset if append else 0
                content_length = response.headers.get("Content-Length")
                expected_size = (
                    starting_size + int(content_length) if content_length else None
                )
                with partial.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
            actual_size = partial.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                raise OSError(
                    f"incomplete response: expected {expected_size} bytes, "
                    f"received {actual_size}"
                )
            os.replace(partial, target)
            return {
                "scene_id": task.scene_id,
                "suffix": task.suffix,
                "path": relative,
                "status": "downloaded",
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "attempts": attempt,
            }
        except Exception as exc:  # noqa: BLE001 - sanitized into the audit manifest
            last_error = _safe_error(exc)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    return {
        "scene_id": task.scene_id,
        "suffix": task.suffix,
        "path": relative,
        "status": "failed",
        "error": last_error,
        "partial_bytes": partial.stat().st_size if partial.exists() else 0,
        "attempts": retries,
    }


def download_scannet_subset(
    *,
    official_downloader: str | Path,
    scene_list: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path,
    accept_tos: bool,
    file_types: tuple[str, ...] = MINIMAL_FILE_TYPES,
    include_label_map: bool = True,
    workers: int = 2,
    retries: int = 3,
    timeout_s: float = 120.0,
    min_free_gb: float = 80.0,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not accept_tos:
        raise ValueError("ScanNet Terms of Use must be accepted explicitly")
    if workers < 1 or retries < 1:
        raise ValueError("workers and retries must be positive")
    if min_free_gb < 0:
        raise ValueError("min_free_gb must be non-negative")

    source_list = Path(scene_list).resolve()
    scene_ids = validate_scene_ids(read_scene_ids(source_list))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive when provided")
        scene_ids = scene_ids[:limit]
    module = load_official_downloader(official_downloader)
    root = Path(out_dir).resolve()
    tasks = build_download_tasks(
        module,
        scene_ids,
        root,
        file_types=file_types,
        include_label_map=include_label_map,
    )
    _check_free_space(root, int(min_free_gb * 1024**3))

    if dry_run:
        files = [
            {
                "scene_id": task.scene_id,
                "suffix": task.suffix,
                "path": task.target.relative_to(root).as_posix(),
                "status": "planned",
            }
            for task in tasks
        ]
    else:
        files = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _download_one,
                    task,
                    root=root,
                    retries=retries,
                    timeout_s=timeout_s,
                    min_free_bytes=int(min_free_gb * 1024**3),
                ): task
                for task in tasks
            }
            for future in as_completed(futures):
                record = future.result()
                files.append(record)
                print(
                    f"[{record['status']}] {record['path']} "
                    f"({record.get('bytes', record.get('partial_bytes', 0))} bytes)",
                    flush=True,
                )
        files.sort(key=lambda item: item["path"])

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "scannet_minimal_download",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tos_accepted": True,
        "official_downloader": Path(official_downloader).name,
        "official_downloader_sha256": sha256_file(official_downloader),
        "scene_list": source_list.name,
        "scene_list_sha256": sha256_file(source_list),
        "scene_count": len(scene_ids),
        "file_types": list(file_types),
        "include_label_map": include_label_map,
        "dry_run": dry_run,
        "files": files,
    }
    payload["content_sha256"] = hash_json(payload)
    write_json(manifest_path, payload)
    failures = [item for item in files if item["status"] == "failed"]
    if failures:
        raise RuntimeError(
            f"ScanNet download failed for {len(failures)} files; see {manifest_path}"
        )
    return payload
