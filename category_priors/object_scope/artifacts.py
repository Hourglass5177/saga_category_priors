"""Immutable, content-addressed storage independent of retired study constants."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from collections.abc import Mapping
import uuid
import numpy as np


def readonly(value, dtype=None):
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if array.dtype.hasobject:
        raise ValueError("object arrays are not evidence")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def array_digest(value):
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("object arrays have no stable content identity")
    header = json.dumps([array.dtype.str, list(array.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def plain(value):
    if dataclasses.is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return {"array_sha256": array_digest(value), "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(plain(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical(value):
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_bytes_once(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(f"immutable output differs: {target}")
        return hashlib.sha256(data).hexdigest()
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != data:
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def write_once(path, value):
    return write_bytes_once(path, canonical(value) + b"\n")


class EvidenceStore:
    """Array files and small metadata are verified on every independent reopen."""
    def __init__(self, root):
        self.root = Path(root)

    def put_array(self, array):
        import io
        value = np.asarray(array)
        key = array_digest(value)
        output = io.BytesIO()
        np.save(output, value, allow_pickle=False)
        relative = f"arrays/{key[:2]}/{key}.npy"
        file_sha = write_bytes_once(self.root / relative, output.getvalue())
        return {"path": relative, "sha256": file_sha, "array_sha256": key,
                "dtype": value.dtype.str, "shape": list(value.shape)}

    def read_array(self, ref):
        path = (self.root / ref["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()) or file_digest(path) != ref["sha256"]:
            raise ValueError("array path/content mismatch")
        array = np.load(path, allow_pickle=False)
        if (array_digest(array) != ref["array_sha256"] or array.dtype.str != ref["dtype"]
                or list(array.shape) != ref["shape"]):
            raise ValueError("array semantic identity mismatch")
        return readonly(array)

    def put(self, identity, result):
        key = digest(identity)
        payload = {"identity": plain(identity), "result": plain(result)}
        write_once(self.root / "records" / key[:2] / (key + ".json"),
                   {**payload, "payload_sha256": digest(payload)})
        return key

    def get(self, identity):
        key = digest(identity)
        row = json.loads((self.root / "records" / key[:2] / (key + ".json")).read_text(encoding="utf8"))
        if canonical(row["identity"]) != canonical(identity) or row["payload_sha256"] != digest({k: row[k] for k in ("identity", "result")}):
            raise ValueError("evidence identity/content mismatch")
        return row["result"]
