"""CPU children only: process completion is never a scientific gate marker."""
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

from category_priors.object_verification import EXPERIMENT_VERSION
from category_priors.object_verification.study import StudyLedger
from category_priors.object_verification import supervisor as sup


@pytest.mark.parametrize('latest_state', ['Z', 'R'])
def test_orphan_empty_environment_rechecks_exit_without_trusting_live_unknown(tmp_path, monkeypatch, latest_state):
    proc = tmp_path/'proc'; (proc/'301').mkdir(parents=True)
    (proc/'301'/'environ').write_bytes(b'')
    boot = tmp_path/'boot'; boot.write_text('test-boot')
    real_path = Path
    monkeypatch.setattr(sup, 'Path', lambda p: proc if str(p)=='/proc' else boot
                        if str(p)=='/proc/sys/kernel/random/boot_id' else real_path(p))
    monkeypatch.setattr(sup.socket, 'gethostname', lambda: 'fixture')
    child = dict(pid=300,host='fixture',scheme='linux-proc-v1',boot_id='test-boot',
                 start_ticks=10,pgid=300,sid=300,state='Z')
    calls=[]
    def identity(pid):
        if pid==300: return child
        calls.append(pid)
        return child | dict(pid=301,start_ticks=11,state='R' if len(calls)==1 else latest_state)
    monkeypatch.setattr(sup, 'process_identity', identity)
    if latest_state=='Z':
        assert sup._linux_members(child,'private-token')==[]
    else:
        with pytest.raises(sup.RecoveryBlocked,match='lacks inherited'):
            sup._linux_members(child,'private-token')


def setup(tmp_path, code, *, kind="cpu", clock=time.time):
    ledger = StudyLedger(tmp_path / "ledger.jsonl", EXPERIMENT_VERSION, clock=clock)
    ledger.initialize(historical_gpu_seconds=24823.804896831512)
    argv = [sys.executable, "-c", code]
    spec = {"experiment_version": EXPERIMENT_VERSION, "kind": kind, "argv": argv,
            "cwd": str(tmp_path.resolve()), "env": {"CUDA_VISIBLE_DEVICES": ""}}
    return ledger, spec


def run(ledger, spec, tmp_path, **kwargs):
    return sup.run_supervised_task(ledger, "fixture", spec=spec, argv=spec["argv"], cwd=spec["cwd"],
        attempt_dir=tmp_path / "attempt-1", env=spec["env"], gpu_count=1 if spec["kind"] == "gpu" else 0,
        gate_validator=lambda row: {"scope": "CPU process fixture only", "model_execution_authorized": False},
        poll_seconds=.01, terminate_grace_seconds=.05, **kwargs)


def test_cpu_success_records_real_receipt_not_scientific_acceptance(tmp_path):
    ledger, spec = setup(tmp_path, "print('CPU fixture')")
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    result = run(ledger, spec, tmp_path)
    assert result["process_complete"] and result["returncode"] == 0
    assert result["scientific_acceptance"] == "not_asserted"
    assert result["ledger"]["completed_gpu_seconds"] == 0
    assert not result["ledger"]["active"]
    assert (tmp_path / "attempt-1" / "stdout.log").read_text().strip() == "CPU fixture"
    assert all(signal.getsignal(s) == h for s, h in handlers.items())
    events = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    record = next(r for r in events if r["event"] == "supervisor_worker")["data"]
    assert record["child_identity"]["host"] == socket.gethostname()
    assert record["child_identity"]["pid"] != os.getpid()
    assert ledger.status()["historical_gpu_seconds_separate"] == 24823.804896831512


def test_failed_child_preserved_and_recovery_needs_new_attempt(tmp_path):
    ledger, spec = setup(tmp_path, "raise SystemExit(7)")
    result = run(ledger, spec, tmp_path)
    assert result["status"] == "failed" and result["returncode"] == 7
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        run(ledger, spec, tmp_path)
    assert ledger.path.read_bytes() == before
    with pytest.raises(ValueError, match="explain"):
        sup.run_supervised_task(ledger, "fixture", spec=spec, argv=spec["argv"], cwd=spec["cwd"],
            attempt_dir=tmp_path / "attempt-2", env=spec["env"], gpu_count=0,
            gate_validator=lambda row: {"scope": "CPU fixture"})
    assert len(ledger.status()["attempts"]) == 1


def test_budget_deadline_kills_real_cpu_child_and_counts_cleanup(tmp_path):
    base = [100.0]
    t0 = time.monotonic()
    clock = lambda: base[0] + time.monotonic() - t0
    ledger, spec = setup(tmp_path, "import time; time.sleep(30)", kind="gpu", clock=clock)
    # This is simulated GPU accounting of CPU-only children, not real GPU use.
    ledger.register_task("previous-fixture", {"experiment_version": EXPERIMENT_VERSION, "kind": "gpu"})
    first = ledger.start_attempt("previous-fixture", owner_identity=sup.process_identity(os.getpid()))
    base[0] += 86400 - .35
    ledger.finish_attempt(first["lease_id"], status="interrupted", reason="synthetic prior occupancy")
    wall = time.monotonic()
    result = run(ledger, spec, tmp_path)
    assert time.monotonic() - wall < 5
    assert result["status"] == "interrupted" and "budget deadline" in result["reason"]
    assert result["returncode"] != 0 and not result["ledger"]["active"]
    assert result["ledger"]["completed_gpu_seconds"] >= 86400
    assert result["ledger"]["remaining_gpu_seconds"] == 0
    assert len(result["ledger"]["attempts"]) == 2


@pytest.mark.parametrize("change", ["argv", "cwd", "env", "gate"])
def test_gate_and_immutable_launch_refuse_before_child(tmp_path, change):
    marker = tmp_path / "must-not-exist"
    ledger, spec = setup(tmp_path, f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')")
    argv, cwd, env = list(spec["argv"]), spec["cwd"], dict(spec["env"])
    if change == "argv": argv[-1] = "print('different')"
    elif change == "cwd": cwd = str(tmp_path.parent)
    elif change == "env": env["CUDA_VISIBLE_DEVICES"] = "0"
    def gate(row):
        if change == "gate": raise ValueError("actual prerequisites pending")
        return {"scope": "CPU test"}
    with pytest.raises(ValueError):
        sup.run_supervised_task(ledger, "fixture", spec=spec, argv=argv, cwd=cwd, env=env,
            attempt_dir=tmp_path / "attempt", gate_validator=gate, gpu_count=0)
    assert not marker.exists() and not ledger.status()["attempts"]


@pytest.mark.parametrize("all_writes", [False, True])
def test_launch_disk_error_settles_stopped_or_unstarted_lease(tmp_path, monkeypatch, all_writes):
    ledger, spec = setup(tmp_path, "print('must not start')", kind="gpu")
    write = sup._write_new
    def broken(path, value):
        if Path(path).name == "launch.json" or all_writes:
            raise OSError("simulated attempt output disk failure")
        return write(path, value)
    monkeypatch.setattr(sup, "_write_new", broken)
    with pytest.raises(OSError, match="disk failure"):
        run(ledger, spec, tmp_path)
    state = ledger.status()
    assert not state["active"] and state["attempts"][0]["status"] == "failed"
    assert not any(row["event"] == "supervisor_worker" for row in ledger._read())


def test_expired_startup_budget_never_releases_real_command(tmp_path):
    marker = tmp_path / "must-not-run"
    ledger, spec = setup(tmp_path, f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')", kind="gpu")
    calls = [0]
    def monotonic():
        calls[0] += 1
        return 0.0 if calls[0] == 1 else 86401.0
    result = run(ledger, spec, tmp_path, monotonic=monotonic)
    assert result["status"] == "interrupted" and "before command gate release" in result["reason"]
    assert not marker.exists() and not ledger.status()["active"]


def test_active_lease_blocks_duplicate_and_live_owner_recovery(tmp_path):
    ledger, spec = setup(tmp_path, "print('not started')", kind="gpu")
    ledger.register_task("fixture", spec)
    ledger.start_attempt("fixture", owner_identity=sup.process_identity(os.getpid()))
    with pytest.raises(ValueError, match="duplicate"):
        run(ledger, spec, tmp_path)
    with pytest.raises(sup.RecoveryBlocked, match="still alive"):
        sup.reconcile_abandoned_lease(ledger)
    assert len(ledger.status()["active"]) == 1


def test_exception_stops_worker_and_restores_signal_handlers(tmp_path):
    ledger, spec = setup(tmp_path, "import time; time.sleep(30)", kind="gpu")
    calls = [0]
    def clock():
        calls[0] += 1
        if calls[0] == 2: raise KeyboardInterrupt("CPU injected controller interruption")
        return time.monotonic()
    old = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        run(ledger, spec, tmp_path, monotonic=clock)
    assert signal.getsignal(signal.SIGTERM) == old
    state = ledger.status()
    assert not state["active"] and state["attempts"][0]["status"] == "interrupted"
    receipt = json.loads((tmp_path / "attempt-1" / "process_result.json").read_text(encoding="utf-8"))
    assert receipt["returncode"] != 0


def test_cpu_background_descendant_is_killed_before_lease_finish(tmp_path):
    marker = tmp_path / "descendant-must-not-finish"
    child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).write_text('leaked')"
    code = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}])"
    ledger, spec = setup(tmp_path, code)
    result = run(ledger, spec, tmp_path)
    assert result["status"] == "failed" and "descendants" in result["reason"]
    time.sleep(1.1)
    assert not marker.exists() and not ledger.status()["active"]


def test_creation_identity_detects_pid_reuse():
    actual = sup.process_identity(os.getpid())
    assert sup.same_process(actual, actual)
    changed = copy.deepcopy(actual)
    field = "start_ticks" if actual["scheme"] == "linux-proc-v1" else "creation_ticks"
    changed[field] += 1
    assert not sup.same_process(changed, actual)
    changed = copy.deepcopy(actual); changed["host"] += "-other"
    assert not sup.same_process(changed, actual)


def test_recovery_refuses_missing_worker_identity_without_reset(tmp_path, monkeypatch):
    ledger, spec = setup(tmp_path, "pass", kind="gpu")
    owner = sup.process_identity(os.getpid())
    ledger.register_task("fixture", spec)
    ledger.start_attempt("fixture", owner_identity=owner)
    monkeypatch.setattr(sup, "process_identity", lambda pid: None)
    raw = ledger.path.read_bytes()
    with pytest.raises(sup.RecoveryBlocked, match="identity missing"):
        sup.reconcile_abandoned_lease(ledger)
    assert ledger.path.read_bytes() == raw and ledger.status()["active"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc recovery needs Linux CPU runner")
def test_linux_reused_child_pid_is_never_signaled(tmp_path, monkeypatch):
    ledger, spec = setup(tmp_path, "pass", kind="gpu")
    owner = sup.process_identity(os.getpid()) | {"token": "fixture"}
    ledger.register_task("fixture", spec)
    lease = ledger.start_attempt("fixture", owner_identity=owner)
    child = copy.deepcopy(owner); child.update(pid=999999, pgid=999999, sid=999999)
    sup._event(ledger, "supervisor_worker", {"lease_id": lease["lease_id"], "child_identity": child, "token": "fixture"})
    monkeypatch.setattr(sup, "process_identity", lambda pid: None if pid == owner["pid"] else child | {"start_ticks": child["start_ticks"] + 1})
    monkeypatch.setattr(os, "killpg", lambda *args: pytest.fail("must not signal reused PID"))
    with pytest.raises(sup.RecoveryBlocked, match="PID reused"):
        sup.reconcile_abandoned_lease(ledger)
    assert ledger.status()["active"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux escaped-session cleanup needs Linux CPU runner")
def test_linux_setsided_descendant_cannot_escape_accounting(tmp_path):
    marker = tmp_path / "escaped-must-not-finish"
    child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).write_text('leak')"
    code = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True)"
    ledger, spec = setup(tmp_path, code)
    result = run(ledger, spec, tmp_path)
    assert result["status"] == "failed"
    assert result["cleanup"]["escaped_pids"]
    time.sleep(1.1)
    assert not marker.exists() and not result["ledger"]["active"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux unreadable /proc cleanup needs Linux CPU runner")
def test_linux_unrelated_unreadable_environment_does_not_spare_known_worker(tmp_path, monkeypatch):
    token = "cpu-permission-fixture"
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True, env=os.environ.copy() | {"SAGA_OBJECT_LEASE_TOKEN": token})
    child = sup.process_identity(process.pid)
    read_bytes = Path.read_bytes
    def denied(path):
        if path == Path(f"/proc/{os.getpid()}/environ"):
            raise PermissionError("CPU fixture unrelated nondumpable process")
        return read_bytes(path)
    monkeypatch.setattr(Path, "read_bytes", denied)
    try:
        # The supervisor existed before its worker, so cannot be its child.
        # Wait a scheduler tick before launching in this fixture if necessary.
        if sup.process_identity(os.getpid())["start_ticks"] == child["start_ticks"]:
            with pytest.raises(sup.RecoveryBlocked, match="cannot inspect"):
                sup._stop_linux(child, token, .05)
        else:
            result = sup._stop_linux(child, token, .05)
            assert process.pid in result["terminated_pids"]
        assert process.wait(timeout=3) != 0
    finally:
        if process.poll() is None:
            process.kill(); process.wait()


def test_dead_owner_real_orphan_recovery_counts_elapsed(tmp_path):
    # The owner dies only after its worker is durably bound and executing.
    script = """import sys,time
from pathlib import Path
from category_priors.object_verification import EXPERIMENT_VERSION
from category_priors.object_verification.study import StudyLedger
from category_priors.object_verification.supervisor import run_supervised_task
root=Path(sys.argv[1]); l=StudyLedger(root/'ledger.jsonl',EXPERIMENT_VERSION); l.initialize()
a=[sys.executable,'-c',"import time; time.sleep(30)"]
s={'experiment_version':EXPERIMENT_VERSION,'kind':'gpu','argv':a,'cwd':str(root),'env':{'CUDA_VISIBLE_DEVICES':''}}
run_supervised_task(l,'orphan',spec=s,argv=a,cwd=root,env=s['env'],attempt_dir=root/'attempt',gpu_count=1,gate_validator=lambda s:{'scope':'CPU-only fixture'})
"""
    owner = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], env=os.environ.copy(),
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    ledger = StudyLedger(tmp_path / "ledger.jsonl", EXPERIMENT_VERSION)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if ledger.path.exists() and b'"event": "supervisor_worker"' in ledger.path.read_bytes(): break
            # Study ledger compact JSON has no spaces.
            if ledger.path.exists() and b'"event":"supervisor_worker"' in ledger.path.read_bytes(): break
            time.sleep(.01)
        else: pytest.fail("worker was not bound")
        owner.kill(); owner.wait(timeout=3)
        time.sleep(.05)
        result = sup.reconcile_abandoned_lease(ledger, "real CPU killed-owner recovery fixture")
        assert not result["ledger"]["active"]
        assert result["ledger"]["completed_gpu_seconds"] > .05
        assert result["ledger"]["attempts"][0]["status"] == "interrupted"
    finally:
        if owner.poll() is None: owner.kill(); owner.wait()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux pidfd syscall fallback")
def test_linux_pidfd_syscall_fallback_real_child(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = sup.process_identity(process.pid)
        fd = sup._pidfd_open(process.pid)
        try:
            assert sup.same_process(identity, sup.process_identity(process.pid))
            sup._pidfd_signal(fd, 0)
            sup._pidfd_signal(fd, signal.SIGKILL)
        finally:
            os.close(fd)
        assert process.wait(timeout=3) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill(); process.wait()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux unknown environment remains fail closed")
def test_linux_potential_escaped_unreadable_process_retains_uncertainty(tmp_path, monkeypatch):
    token = "cpu-unknown-fixture"
    root = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True, env=os.environ.copy() | {"SAGA_OBJECT_LEASE_TOKEN": token})
    time.sleep(.03)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],start_new_session=True)
    child = sup.process_identity(root.pid)
    original = Path.read_bytes
    def denied(path):
        if path == Path(f"/proc/{unrelated.pid}/environ"):
            raise PermissionError("unknown newer process")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", denied)
    try:
        with pytest.raises(sup.RecoveryBlocked, match="potentially owned"):
            sup._stop_linux(child, token, .05)
        assert root.wait(timeout=3) != 0
        assert unrelated.poll() is None  # No evidence authorizes signaling it.
    finally:
        for process in (root, unrelated):
            if process.poll() is None:
                process.kill(); process.wait()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux process identity must bind before signal")
def test_linux_pidfd_open_identity_race_never_signals(monkeypatch):
    child = sup.process_identity(os.getpid()) | {"pid":999999,"pgid":999999,"sid":999999}
    monkeypatch.setattr(sup,"_linux_members",lambda *args:[child])
    monkeypatch.setattr(sup,"_pidfd_open",lambda pid:42)
    monkeypatch.setattr(sup,"process_identity",lambda pid:child | {"start_ticks":child["start_ticks"]+1})
    monkeypatch.setattr(sup,"_pidfd_signal",lambda *args:pytest.fail("reused PID must not be signaled"))
    monkeypatch.setattr(os,"close",lambda fd:None)
    with pytest.raises(sup.RecoveryBlocked,match="PID reused"):
        sup._stop_linux(child,"fake-token",0)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux termination capability gate")
def test_linux_pidfd_unavailable_refuses_before_lease_or_command(tmp_path, monkeypatch):
    marker = tmp_path / "must-not-run"
    ledger, spec = setup(tmp_path, f"from pathlib import Path; Path({str(marker)!r}).touch()")
    def denied(pid):
        raise PermissionError("simulated seccomp pidfd denial")
    monkeypatch.setattr(sup, "_pidfd_open", denied)
    with pytest.raises(PermissionError, match="seccomp"):
        run(ledger, spec, tmp_path)
    assert not marker.exists() and not ledger.status()["attempts"]
