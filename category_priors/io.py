from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    target.write_text(payload + "\n", encoding="utf-8")


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("Reading parquet requires pandas and pyarrow") from exc
        return pd.read_parquet(source).to_dict(orient="records")
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{source}:{line_number}: expected an object")
                rows.append(value)
        return rows
    if suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported table format: {source.suffix}")


def write_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    normalized = [dict(row) for row in rows]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("Writing parquet requires pandas and pyarrow") from exc
        pd.DataFrame(normalized).to_parquet(target, index=False)
        return
    if suffix == ".jsonl":
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for row in normalized:
                handle.write(canonical_json_bytes(row).decode("utf-8") + "\n")
        return
    if suffix == ".csv":
        fieldnames = sorted({key for row in normalized for key in row})
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized)
        return
    raise ValueError(f"Unsupported table format: {target.suffix}")


def build_file_manifest(
    paths: Iterable[str | Path], root: str | Path | None = None
) -> list[dict[str, Any]]:
    base = Path(root).resolve() if root is not None else None
    manifest: list[dict[str, Any]] = []
    for raw_path in sorted((Path(path).resolve() for path in paths), key=str):
        try:
            display_path = str(raw_path.relative_to(base)) if base else str(raw_path)
        except ValueError:
            display_path = str(raw_path)
        manifest.append(
            {
                "path": display_path.replace("\\", "/"),
                "size": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            }
        )
    return manifest
