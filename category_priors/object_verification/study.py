"""Cumulative GPU accounting shared by the experiment runner and supervisor."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

TOTAL_GPU_SECONDS = 86400


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _number(value: Any, *, nonnegative: bool = True) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and (not nonnegative or value >= 0)


def validate_evidence(references: Sequence[Mapping[str, Any]], *, verify_files: bool = True, allow_empty: bool = False) -> list[dict[str, Any]]:
    """Validate immutable file references {path:absolute, sha256:lowercase hex}."""
    if not isinstance(references, (list, tuple)) or (not references and not allow_empty):
        raise ValueError("nonempty evidence references are required")
    result, seen = [], set()
    for reference in references:
        ref = _clone(reference)
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not _sha(ref.get("sha256")):
            raise ValueError("invalid evidence reference")
        if ref["path"] in seen:
            raise ValueError("duplicate evidence path")
        seen.add(ref["path"])
        if verify_files:
            path = Path(ref["path"])
            if not path.is_absolute() or not path.is_file():
                raise ValueError("evidence must exist at an absolute file path")
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != ref["sha256"]:
                raise ValueError("evidence hash mismatch")
        result.append(ref)
    return result


class StudyLedger:
    """Hash-linked JSONL: immutable task specs, new attempts, cumulative GPU use.

    This is accounting, not a process supervisor. The controller must terminate
    only its own worker before recording interruption; abandoned active leases
    continue counting occupancy in status(). No method clears an active lease
    or silently discards elapsed use. Times are injectable for CPU tests.
    """
    def __init__(self, path: str | Path, experiment_version: str, *, clock: Callable[[], float] = time.time):
        self.path = Path(path).resolve()
        if not isinstance(experiment_version, str) or not experiment_version:
            raise ValueError("new experiment version is mandatory")
        self.experiment_version, self.clock = experiment_version, clock

    @contextlib.contextmanager
    def _lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        path = self.path.with_suffix(self.path.suffix + ".lock")
        with path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt
                handle.seek(0, 2)
                if handle.tell() == 0:
                    handle.write(b"0"); handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("ledger has a partial event; preserve and inspect it")
        rows, previous = [], "0" * 64
        for number, line in enumerate(raw.splitlines()):
            row = json.loads(line)
            digest = row.pop("sha256", None)
            if row.get("sequence") != number or row.get("previous_sha256") != previous or _digest(row) != digest:
                raise ValueError("ledger event chain was changed or truncated")
            row["sha256"] = digest
            if row.get("experiment_version") != self.experiment_version or not _number(row.get("at")):
                raise ValueError("ledger version/time differs")
            if rows and row["at"] < rows[-1]["at"]:
                raise ValueError("ledger time moved backwards")
            rows.append(row); previous = digest
        if rows and (rows[0]["event"] != "initialize" or rows[0]["data"].get("total_cap_seconds") != TOTAL_GPU_SECONDS):
            raise ValueError("ledger must be the independent registered 86400-second study")
        return rows

    def _append(self, rows: list[dict[str, Any]], event: str, data: Mapping[str, Any], *, at: float | None = None) -> None:
        now = float(self.clock()) if at is None else at
        if not _number(now) or (rows and now < rows[-1]["at"]):
            raise ValueError("accounting clock moved backwards")
        row = {"sequence": len(rows), "previous_sha256": rows[-1]["sha256"] if rows else "0" * 64,
               "experiment_version": self.experiment_version, "at": now, "event": event, "data": _clone(data)}
        row["sha256"] = _digest(row)
        with self.path.open("ab") as handle:
            handle.write(_canonical(row) + b"\n"); handle.flush(); os.fsync(handle.fileno())
        rows.append(row)

    def initialize(self, *, historical_gpu_seconds: float | None = None, reason: str = "user-authorized independent object-verification stage") -> dict[str, Any]:
        if (historical_gpu_seconds is not None and not _number(historical_gpu_seconds)) or not str(reason).strip():
            raise ValueError("historical display and authorization reason must be valid")
        with self._lock():
            rows = self._read()
            if rows:
                raise ValueError("ledger already initialized; it cannot be reset")
            self._append(rows, "initialize", {"total_cap_seconds": TOTAL_GPU_SECONDS, "historical_gpu_seconds_separate": historical_gpu_seconds, "reason": reason})
            return self._state(rows)

    def _state(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        tasks, attempts, active = {}, {}, {}
        used = 0.0
        for row in rows:
            data, kind = row["data"], row["event"]
            if kind == "register_task":
                tasks[data["task_id"]] = data
            elif kind == "start_attempt":
                value = _clone(data) | {"started_at": row["at"], "status": "running"}
                attempts[data["lease_id"]] = value; active[data["lease_id"]] = value
            elif kind == "finish_attempt":
                value = attempts[data["lease_id"]]
                seconds = (row["at"] - value["started_at"]) * value["gpu_count"]
                if not math.isclose(seconds, data["occupied_gpu_seconds"], abs_tol=1e-9):
                    raise ValueError("recorded occupancy differs from immutable start/finish times")
                used += seconds
                value.update(_clone(data)); value["finished_at"] = row["at"]
                active.pop(data["lease_id"])
        now = float(self.clock())
        if rows and (not _number(now) or now < rows[-1]["at"]):
            raise ValueError("clock moved backwards")
        occupied = sum((now - r["started_at"]) * r["gpu_count"] for r in active.values())
        return {"schema": "saga-object-study-ledger-v1", "experiment_version": self.experiment_version,
                "initialized": bool(rows), "total_cap_seconds": TOTAL_GPU_SECONDS,
                "historical_gpu_seconds_separate": rows[0]["data"]["historical_gpu_seconds_separate"] if rows else None,
                "completed_gpu_seconds": used, "active_gpu_seconds": occupied,
                "remaining_gpu_seconds": max(0.0, TOTAL_GPU_SECONDS - used - occupied),
                "tasks": tasks, "attempts": list(attempts.values()), "active": list(active.values()),
                "last_event_sha256": rows[-1]["sha256"] if rows else None}

    def status(self) -> dict[str, Any]:
        with self._lock():
            return self._state(self._read())

    def register_task(self, task_id: str, spec: Mapping[str, Any]) -> dict[str, Any]:
        value = _clone(spec)
        if not isinstance(task_id, str) or not task_id or value.get("experiment_version") != self.experiment_version or value.get("kind") not in {"cpu", "gpu"}:
            raise ValueError("task needs its immutable version and cpu/gpu kind")
        with self._lock():
            rows = self._read(); state = self._state(rows)
            if not rows:
                raise ValueError("initialize the new ledger explicitly")
            prior = state["tasks"].get(task_id)
            if prior:
                if prior["spec_sha256"] != _digest(value):
                    raise ValueError("task spec is immutable; a science change needs a new version")
                return _clone(prior)
            data = {"task_id": task_id, "spec": value, "spec_sha256": _digest(value)}
            self._append(rows, "register_task", data)
            return _clone(data)

    def start_attempt(self, task_id: str, *, gpu_count: int = 1, recovery_reason: str | None = None, owner_identity: Mapping[str, Any]) -> dict[str, Any]:
        if type(gpu_count) is not int or gpu_count < 0 or not isinstance(owner_identity, Mapping) or not owner_identity:
            raise ValueError("GPU count and explicit supervisor identity are required")
        with self._lock():
            rows = self._read(); state = self._state(rows)
            if task_id not in state["tasks"]:
                raise ValueError("register task before starting")
            task = state["tasks"][task_id]
            if (task["spec"]["kind"] == "gpu") != (gpu_count > 0):
                raise ValueError("actual GPU lease differs from registered task kind")
            previous = [r for r in state["attempts"] if r["task_id"] == task_id]
            if any(r["status"] in {"running", "complete"} for r in previous):
                raise ValueError("do not duplicate a running or completed task")
            if previous and not str(recovery_reason or "").strip():
                raise ValueError("a new attempt must preserve and explain the prior failure")
            if gpu_count and (state["remaining_gpu_seconds"] <= 0 or any(r["gpu_count"] for r in state["active"])):
                raise ValueError("GPU budget exhausted or another GPU lease is active")
            attempt = len(previous) + 1
            lease = _digest({"task": task_id, "attempt": attempt, "previous": state["last_event_sha256"]})
            data = {"task_id": task_id, "attempt": attempt, "lease_id": lease, "gpu_count": gpu_count,
                    "owner_identity": _clone(owner_identity), "recovery_reason": recovery_reason,
                    "spec_sha256": task["spec_sha256"]}
            self._append(rows, "start_attempt", data)
            return _clone(data) | {"max_wall_seconds": state["remaining_gpu_seconds"] / gpu_count if gpu_count else None}

    def finish_attempt(self, lease_id: str, *, status: str, reason: str | None = None, outputs: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
        if status not in {"complete", "failed", "interrupted"} or (status != "complete" and not str(reason or "").strip()):
            raise ValueError("finish status and failure/interruption reason are required")
        refs = validate_evidence(outputs, allow_empty=status != "complete")
        with self._lock():
            rows = self._read(); state = self._state(rows)
            matches = [r for r in state["active"] if r["lease_id"] == lease_id]
            if len(matches) != 1:
                raise ValueError("only the exact active lease can finish")
            active = matches[0]
            ended_at = float(self.clock())
            elapsed = (ended_at - active["started_at"]) * active["gpu_count"]
            if elapsed < 0:
                raise ValueError("clock moved backwards")
            self._append(rows, "finish_attempt", {"lease_id": lease_id, "status": status, "reason": reason,
                "occupied_gpu_seconds": elapsed, "outputs": refs}, at=ended_at)
            return self._state(rows)
