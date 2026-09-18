"""Process ownership and hard deadlines for the new independent study ledger.

No model is imported. A zero exit status means only that the command exited;
the caller must independently validate scientific outputs. ``gate_validator``
must raise on absent/invalid real prerequisites and returns its audit record.
The ledger's hash-linked events bind worker identity before its stdin gate is
released. An abandoned/ambiguous process never silently releases its lease.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable, Mapping, Sequence

from .study import StudyLedger


class RecoveryBlocked(RuntimeError):
    """Process ownership or termination cannot be proved; retain accounting."""


class SupervisorInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"supervisor received signal {signum}")


class _DeadlineReached(Exception):
    pass


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_json_bytes(value)); stream.flush(); os.fsync(stream.fileno())
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _event(ledger, kind, data):
    # Additional immutable audit events; StudyLedger accounting ignores these.
    with ledger._lock():
        rows = ledger._read()
        ledger._append(rows, kind, data)


def process_identity(pid: int) -> dict[str, Any] | None:
    """Kernel creation identity. Permission errors are not evidence of death."""
    base = {"pid": int(pid), "host": socket.gethostname()}
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{int(pid)}/stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return None
        fields = stat[stat.rfind(")") + 2:].split()
        return base | {"scheme": "linux-proc-v1", "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                       "start_ticks": int(fields[19]), "state": fields[0],
                       "pgid": int(fields[2]), "sid": int(fields[3])}
    if os.name == "nt":
        from ctypes import wintypes as w
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]; kernel.OpenProcess.restype = w.HANDLE
        kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
        kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            if ctypes.get_last_error() == 87:
                return None
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            times = [w.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                raise ctypes.WinError(ctypes.get_last_error())
            code = w.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return base | {"scheme": "windows-creation-v1", "creation_ticks": (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime,
                           "state": "running" if code.value == 259 else "exited"}
        finally:
            kernel.CloseHandle(handle)
    raise RecoveryBlocked("kernel-backed supervision supports Linux and Windows only")


def same_process(expected, actual):
    keys = ("scheme", "host", "pid", "boot_id", "start_ticks") if expected.get("scheme") == "linux-proc-v1" else ("scheme", "host", "pid", "creation_ticks")
    return actual is not None and all(expected.get(k) == actual.get(k) for k in keys)


def _linux_members(child, token):
    if child.get("host") != socket.gethostname() or child.get("pgid") != child["pid"] or child.get("sid") != child["pid"]:
        raise RecoveryBlocked("worker private session/host identity differs")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != child["boot_id"]:
        return []  # A completed reboot proves previous processes are gone.
    actual = process_identity(child["pid"])
    if actual is not None and not same_process(child, actual):
        raise RecoveryBlocked("worker PID reused; no process was signaled")
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        row = process_identity(int(path.name))
        if row is None or row["pgid"] != child["pgid"] or row["state"] in {"Z", "X"}:
            continue
        if row["sid"] != child["sid"] or row["start_ticks"] < child["start_ticks"]:
            raise RecoveryBlocked("worker group membership cannot be proved")
        if actual is None or actual["state"] in {"Z", "X"}:
            try:
                environment = (path / "environ").read_bytes().split(b"\0")
            except (FileNotFoundError, ProcessLookupError):
                continue
            if ("SAGA_OBJECT_LEASE_TOKEN=" + token).encode() not in environment:
                # SIGKILL can clear environ after the preceding stat read still
                # reported a live task. Recheck kernel state before calling that
                # exit race an unowned orphan; live/reused PIDs still fail closed.
                latest = process_identity(row['pid'])
                if latest is None or (same_process(row, latest) and latest['state'] in {'Z', 'X'}):
                    continue
                raise RecoveryBlocked("orphan group member lacks inherited lease token")
        members.append(row)
    return members


def _pidfd_open(pid):
    """Bind a kernel task even on Python/libc builds without pidfd wrappers.

    These syscall numbers are shared by Linux x86_64 and aarch64 only. Unknown
    architectures fail closed; there is no PID-based signaling fallback.
    """
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    if not sys.platform.startswith("linux") or platform.machine() not in {"x86_64", "aarch64"}:
        raise RecoveryBlocked("pidfd syscall ABI is not validated on this architecture")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(fd)


def _pidfd_signal(fd, sig):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(fd, sig)
    if not sys.platform.startswith("linux") or platform.machine() not in {"x86_64", "aarch64"}:
        raise RecoveryBlocked("pidfd signal syscall ABI is not validated on this architecture")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(424), ctypes.c_int(fd), ctypes.c_int(sig),
                          ctypes.c_void_p(), ctypes.c_uint(0))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _stop_linux(child, token, grace):
    def owned():
        group = _linux_members(child, token)
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != child["boot_id"]:
            return []
        indexed = {r["pid"]: r for r in group}
        # A normal child may call setsid(). It still inherits the unique lease
        # token and cannot be omitted merely because it left our original pgid.
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            row = process_identity(int(path.name))
            if row is None or row["state"] in {"Z", "X"}:
                continue
            # A descendant cannot predate its parent. Equal scheduler ticks
            # remain ambiguous and are inspected, never presumed unrelated.
            if row["start_ticks"] < child["start_ticks"]:
                continue
            try:
                environment = (path / "environ").read_bytes().split(b"\0")
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError as exc:
                # /proc environ can become unreadable as a process exits.
                # Only kernel-confirmed death discharges this uncertainty.
                latest = process_identity(row["pid"])
                if latest is None or (same_process(row, latest) and latest["state"] in {"Z", "X"}):
                    continue
                raise RecoveryBlocked("cannot inspect potentially owned process environment for escaped workers") from exc
            if ("SAGA_OBJECT_LEASE_TOKEN=" + token).encode() in environment:
                if row["start_ticks"] < child["start_ticks"]:
                    raise RecoveryBlocked("lease token belongs to a process predating this worker")
                indexed[row["pid"]] = row
        return list(indexed.values())
    def signal_owned(rows, sig):
        for row in rows:
            # pidfd binds the signal to a kernel process, closing the PID reuse
            # race between last identity read and signaling an escaped child.
            try:
                fd = _pidfd_open(row["pid"])
            except ProcessLookupError:
                continue
            try:
                latest = process_identity(row["pid"])
                if latest is None:
                    continue
                if not same_process(row, latest):
                    raise RecoveryBlocked("descendant PID reused; no signal sent")
                _pidfd_signal(fd, sig)
            except ProcessLookupError:
                pass
            finally:
                os.close(fd)
    # Stop the already verified private group *before* a full /proc sweep.
    # An unrelated nondumpable process must not prevent hard-stopping a known
    # GPU worker. A later sweep failure still retains the active lease because
    # escaped descendants have not yet been excluded.
    known = _linux_members(child, token)
    if known:
        signal_owned(known, signal.SIGKILL)
    initial = owned()
    if not initial:
        return {"terminated_pids": [r["pid"] for r in known], "escaped_pids": [],
                "cleanup": "verified_group_and_token_descendants_stopped" if known else "already_stopped"}
    stages = ((signal.SIGTERM, grace), (signal.SIGKILL, 3.0)) if grace else ((signal.SIGKILL, 3.0),)
    for sig, timeout in stages:
        members = owned()
        if not members:
            break
        # Per-task pidfds avoid an unbound killpg PID-reuse race. Repeated
        # scans include descendants forked before their parent was stopped.
        signal_owned(members, sig)
        deadline = time.monotonic() + timeout
        while owned() and time.monotonic() < deadline:
            # A child can fork during SIGTERM cleanup; repeat the bound sweep.
            if sig == signal.SIGKILL:
                signal_owned(owned(), sig)
            time.sleep(.01)
    if owned():
        raise RecoveryBlocked("owned process group still alive; lease remains active")
    return {"terminated_pids": sorted({r["pid"] for r in known + initial}), "escaped_pids": [r["pid"] for r in initial if r["pgid"] != child["pgid"]],
            "cleanup": "verified_group_and_token_descendants_stopped"}


class _WindowsJob:
    """Named, non-inherited kill-on-close Job owns the gated wrapper and children."""
    def __init__(self, name, *, open_existing=False):
        from ctypes import wintypes as w
        self.k = ctypes.WinDLL("kernel32", use_last_error=True)
        for method, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            ("OpenJobObjectW", [w.DWORD, w.BOOL, w.LPCWSTR], w.HANDLE),
            ("SetInformationJobObject", [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            ("AssignProcessToJobObject", [w.HANDLE, w.HANDLE], w.BOOL),
            ("TerminateJobObject", [w.HANDLE, w.UINT], w.BOOL),
            ("QueryInformationJobObject", [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
            ("CloseHandle", [w.HANDLE], w.BOOL)):
            fn = getattr(self.k, method); fn.argtypes = args; fn.restype = result
        self.handle = self.k.OpenJobObjectW(0x1F003F, False, name) if open_existing else self.k.CreateJobObjectW(None, name)
        if not self.handle:
            error = ctypes.get_last_error()
            if open_existing and error == 2:
                return
            raise ctypes.WinError(error)
        if not open_existing:
            class Basic(ctypes.Structure):
                _fields_ = [("per_process", ctypes.c_longlong), ("per_job", ctypes.c_longlong), ("flags", w.DWORD),
                            ("min_work", ctypes.c_size_t), ("max_work", ctypes.c_size_t), ("active", w.DWORD),
                            ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]
            class Extended(ctypes.Structure):
                _fields_ = [("basic", Basic), ("io", ctypes.c_ulonglong * 6), ("process_memory", ctypes.c_size_t),
                            ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
            limits = Extended(); limits.basic.flags = 0x2000
            if not self.k.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                self.close(); raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        if not self.k.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def active_count(self):
        if not self.handle:
            return 0
        from ctypes import wintypes as w
        class Accounting(ctypes.Structure):
            _fields_ = [("times", ctypes.c_longlong * 4), ("faults", w.DWORD), ("total", w.DWORD),
                        ("active", w.DWORD), ("terminated", w.DWORD)]
        data = Accounting()
        if not self.k.QueryInformationJobObject(self.handle, 1, ctypes.byref(data), ctypes.sizeof(data), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return data.active

    def stop(self):
        if self.handle and self.active_count():
            if not self.k.TerminateJobObject(self.handle, 2):
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic() + 3
            while self.active_count() and time.monotonic() < deadline:
                time.sleep(.01)
            if self.active_count():
                raise RecoveryBlocked("owned Windows job remains active")
        return {"cleanup": "verified_job_stopped"}

    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle); self.handle = None


_WORKER = """import json,os,subprocess,sys
line=sys.stdin.buffer.readline()
if not line: sys.exit(125)
p=json.loads(line)
os.chdir(p['cwd'])
if os.name=='posix': os.execvpe(p['argv'][0],p['argv'],os.environ)
sys.exit(subprocess.call(p['argv'],shell=False,stdin=subprocess.DEVNULL,stdout=sys.stdout,stderr=sys.stderr,creationflags=subprocess.CREATE_NO_WINDOW))
"""


def run_supervised_task(ledger: StudyLedger, task_id: str, *, spec: Mapping[str, Any], argv: Sequence[str],
                        cwd: str | Path, attempt_dir: str | Path, gate_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                        gpu_count: int = 1, recovery_reason: str | None = None, env: Mapping[str, str] | None = None,
                        poll_seconds: float = .05, terminate_grace_seconds: float = 1.0,
                        monotonic: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Run explicit argv under the remaining ledger budget, without shell parsing.

    Immutable ``spec`` must contain identical argv/cwd/env. The attempt directory
    must not exist; every attempt retains its own logs/receipt. Gate callback is
    mandatory and owns actual prerequisite validation; this utility never makes
    scientific acceptance or GPU-success claims from a process exit code.
    """
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("supervisor requires main thread for scoped signal handling")
    if isinstance(argv, (str, bytes)) or not argv or any(not isinstance(s, str) or not s or "\0" in s for s in argv):
        raise ValueError("explicit nonempty argv list is required")
    cwd = Path(cwd).resolve(); env = dict(env or {}); argv = list(argv)
    if not cwd.is_dir() or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError("valid working directory/environment overrides required")
    if spec.get("argv") != argv or spec.get("cwd") != str(cwd) or spec.get("env") != env:
        raise ValueError("actual command/cwd/environment differ from immutable task spec")
    if not 0 < poll_seconds <= 1 or not 0 <= terminate_grace_seconds <= 10:
        raise ValueError("bounded supervision polling/termination grace required")
    frozen_spec = _json_bytes(spec)
    gates = gate_validator(json.loads(frozen_spec))
    if not isinstance(gates, Mapping) or not gates or _json_bytes(spec) != frozen_spec:
        raise ValueError("gate validator must return an audit and cannot mutate frozen spec")
    if sys.platform.startswith("linux"):
        # Prove the termination primitive before acquiring a lease or releasing
        # any command; a missing syscall/seccomp denial cannot strand a job.
        probe_fd = _pidfd_open(os.getpid())
        try:
            _pidfd_signal(probe_fd, 0)
        finally:
            os.close(probe_fd)
    owner = process_identity(os.getpid())
    if owner is None:
        raise RecoveryBlocked("cannot identify current supervisor")
    directory = Path(attempt_dir).resolve()
    if directory.exists():
        raise ValueError("attempt directory already exists; preserve previous attempt")
    ledger.register_task(task_id, spec)
    directory.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    started = monotonic()
    lease = ledger.start_attempt(task_id, gpu_count=gpu_count, recovery_reason=recovery_reason,
                                 owner_identity=owner | {"attempt_dir": str(directory), "token": token})
    deadline = None if lease["max_wall_seconds"] is None else started + lease["max_wall_seconds"]
    process = None; child = None; job = None; cleanup = None; error = None
    status = "failed"; reason = "command did not start"; returncode = None
    handled = [signal.SIGTERM, signal.SIGINT] + ([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])
    previous = {sig: signal.getsignal(sig) for sig in handled}
    def interrupt(signum, frame):
        raise SupervisorInterrupted(signum)
    try:
        for sig in handled:
            signal.signal(sig, interrupt)
        _write_new(directory / "launch.json", {"schema": "saga-object-supervisor-launch-v1", "spec": json.loads(frozen_spec),
                   "lease": lease, "gates": gates, "scope": "process supervision; scientific acceptance is separate"})
        actual_env = os.environ.copy(); actual_env.update(env); actual_env["SAGA_OBJECT_LEASE_TOKEN"] = token
        with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable, "-c", _WORKER], cwd=cwd, env=actual_env,
                                       stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, shell=False,
                                       start_new_session=os.name == "posix",
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if os.name == "nt":
                job = _WindowsJob("Local\\SAGAObject-" + token); job.assign(process)
            child = process_identity(process.pid)
            if child is None:
                raise RecoveryBlocked("gated worker disappeared before identity capture")
            _event(ledger, "supervisor_worker", {"lease_id": lease["lease_id"], "child_identity": child,
                   "job_name": "Local\\SAGAObject-" + token if job else None, "token": token})
            # Hash-linked identity is durable before any user command can run.
            if deadline is not None and monotonic() >= deadline:
                raise _DeadlineReached("budget exhausted before command gate release")
            process.stdin.write(_json_bytes({"argv": argv, "cwd": str(cwd)}) + b"\n")
            process.stdin.close()
            while process.poll() is None:
                if deadline is not None and monotonic() >= deadline:
                    status, reason = "interrupted", "remaining GPU occupancy budget deadline reached"
                    break
                time.sleep(poll_seconds)
            else:
                returncode = process.returncode
                status = "complete" if returncode == 0 else "failed"
                reason = None if returncode == 0 else f"command exited {returncode}"
                if deadline is not None and monotonic() >= deadline:
                    raise _DeadlineReached("command completion observed after budget deadline")
    except _DeadlineReached as exc:
        status, reason = "interrupted", str(exc)
    except BaseException as exc:
        error = exc
        status = "interrupted" if isinstance(exc, (SupervisorInterrupted, KeyboardInterrupt)) else "failed"
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        # Further handled signals cannot interrupt owned-group shutdown/accounting.
        for sig in handled:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if process is not None:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()  # An unreleased worker cannot start argv.
                if child is not None:
                    lingering = job.active_count() if job else len(_linux_members(child, token))
                    if status == "complete" and lingering:
                        status, reason = "failed", "command left live descendants; terminated before lease closure"
                    exhausted = deadline is not None and time.monotonic() >= deadline
                    # Budget expiry gets immediate hard termination. Kernel
                    # scheduling/cleanup latency is still recorded, never hidden.
                    cleanup = job.stop() if job else _stop_linux(child, token, 0.0 if exhausted or "budget" in str(reason) else terminate_grace_seconds)
                    if status == "complete" and cleanup.get("terminated_pids"):
                        status, reason = "failed", "command left escaped descendants; terminated before lease closure"
                elif process.poll() is None:
                    # Not released: this is the exact Popen handle, never a PID lookup.
                    process.kill()
                process.wait(timeout=5)
                returncode = process.returncode
            receipt = {"schema": "saga-object-process-result-v1", "task_id": task_id, "lease_id": lease["lease_id"],
                       "status": status, "process_complete": status == "complete", "scientific_acceptance": "not_asserted",
                       "returncode": returncode, "reason": reason, "child_identity": child, "cleanup": cleanup}
            try:
                output = _write_new(directory / "process_result.json", receipt)
            except Exception as receipt_error:
                # A failed output disk must not strand a stopped/no-worker lease.
                # Preserve the error in the separate append-only accounting log.
                output = None; status = "failed"
                reason = f"receipt persistence failed after owned processes stopped: {receipt_error}; original={reason}"
                _event(ledger, "supervisor_receipt_error", {"lease_id": lease["lease_id"], "reason": reason})
                error = error or receipt_error
            ledger.finish_attempt(lease["lease_id"], status=status, reason=reason, outputs=[output] if output else [])
        except BaseException as cleanup_error:
            # Do not close a lease whose group might still occupy the device.
            _event(ledger, "supervisor_cleanup_blocked", {"lease_id": lease["lease_id"], "error": repr(cleanup_error), "original_error": repr(error)})
            raise RecoveryBlocked(f"termination/accounting incomplete; active lease retained: {cleanup_error}") from cleanup_error
        finally:
            if job:
                job.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if error is not None:
        raise error
    return receipt | {"receipt": output, "ledger": ledger.status()}


def reconcile_abandoned_lease(ledger: StudyLedger, reason: str = "supervisor disappeared; controller resumed") -> dict[str, Any]:
    """Clean only a proved dead owner's recorded worker; never reset old use."""
    if not str(reason).strip():
        raise ValueError("recovery reason is required")
    # Hold ledger lock throughout identity checks/termination to exclude a second
    # controller from racing recovery. Finish accounting after releasing this lock.
    with ledger._lock():
        rows = ledger._read(); state = ledger._state(rows)
        if len(state["active"]) != 1:
            raise RecoveryBlocked("recovery requires exactly one explicitly identified active lease")
        active = state["active"][0]; owner = active["owner_identity"]
        if owner.get("host") != socket.gethostname():
            raise RecoveryBlocked("owner host differs; remote death cannot be inferred")
        actual = process_identity(owner["pid"])
        if same_process(owner, actual) and actual.get("state") not in {"Z", "X", "exited"}:
            raise RecoveryBlocked("supervisor is still alive; no duplicate start or cleanup")
        records = [row["data"] for row in rows if row["event"] == "supervisor_worker" and row["data"]["lease_id"] == active["lease_id"]]
        if len(records) != 1:
            raise RecoveryBlocked("worker creation identity missing/ambiguous; preserve active lease")
        record = records[0]; child = record["child_identity"]
        if child.get("host") != socket.gethostname() or record["token"] != owner.get("token"):
            raise RecoveryBlocked("worker/owner identity binding differs")
        if child["scheme"] == "linux-proc-v1":
            cleanup = _stop_linux(child, record["token"], .2)
        elif child["scheme"] == "windows-creation-v1":
            actual_child = process_identity(child["pid"])
            if actual_child is not None and not same_process(child, actual_child):
                raise RecoveryBlocked("worker PID reused; no signaling or lease reset")
            job = _WindowsJob(record["job_name"], open_existing=True)
            try:
                if not job.handle and same_process(child, actual_child) and actual_child["state"] == "running":
                    raise RecoveryBlocked("recorded worker alive but job identity unavailable")
                cleanup = job.stop()
            finally:
                job.close()
        else:
            raise RecoveryBlocked("unknown kernel process identity")
        ledger._append(rows, "supervisor_recovered", {"lease_id": active["lease_id"], "reason": reason, "cleanup": cleanup})
    # Another start still sees this active lease and fails closed until finish.
    ledger.finish_attempt(active["lease_id"], status="interrupted", reason=reason)
    return {"schema": "saga-object-supervisor-recovery-v1", "lease_id": active["lease_id"], "cleanup": cleanup,
            "ledger": ledger.status()}
