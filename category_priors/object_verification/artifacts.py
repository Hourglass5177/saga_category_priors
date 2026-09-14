"""Content-addressed, role-bound evidence storage; CPU only.

Files are immutable. A missing or corrupt cache is an error, never an empty
scientific observation. A measured empty observation is an ordinary saved record.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import EXPERIMENT_VERSION, SCHEMA


def plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(plain(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_once(path: str | Path, value: Any) -> str:
    """Publish via atomic link; neither retries nor concurrent writers overwrite."""
    import uuid
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value) + b"\n"
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(f"immutable artifact differs: {target}")
        return hashlib.sha256(data).hexdigest()
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != data:
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def cache_identity(*, scene: str, condition: str, role: str, camera: str,
                   inputs: Mapping[str, Any], backend: Mapping[str, Any],
                   algorithm_sha: str, config_sha: str) -> dict[str, Any]:
    required = {"backend", "module_sha256", "binary_sha256"}
    if not required <= backend.keys() or backend["backend"] != "gradient-reference":
        raise ValueError("freeze the actual gradient-reference module and binary identity")
    if any(not backend[key] for key in required) or not algorithm_sha or not config_sha:
        raise ValueError("cache cannot use unknown implementation identity")
    if role not in {"prepass", "construction", "online_verification", "offline_evaluation"}:
        raise ValueError("explicit evidence role required")
    return dict(schema=SCHEMA, experiment_version=EXPERIMENT_VERSION, scene=scene,
                condition=condition, role=role, camera=camera, inputs=plain(inputs),
                backend=plain(backend), algorithm_sha=algorithm_sha, config_sha=config_sha)


class EvidenceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put(self, identity: Mapping[str, Any], result: Mapping[str, Any]) -> str:
        key = digest(identity)
        payload = {"identity": plain(identity), "result": plain(result)}
        write_once(self.root / key[:2] / (key + ".json"),
                   {**payload, "payload_sha256": digest(payload)})
        return key

    def get(self, identity: Mapping[str, Any]) -> dict[str, Any]:
        key = digest(identity)
        path = self.root / key[:2] / (key + ".json")
        row = json.loads(path.read_text(encoding="utf-8"))
        if canonical(row.get("identity")) != canonical(identity):
            raise ValueError("cache input identity mismatch")
        if row.get("payload_sha256") != digest({"identity": row["identity"], "result": row["result"]}):
            raise ValueError("cache content mismatch")
        return row["result"]


def source_inventory(root: str | Path) -> dict[str, str]:
    base = Path(root)
    return {path.relative_to(base).as_posix(): file_digest(path)
            for path in sorted(base.rglob("*.py")) if "__pycache__" not in path.parts}
