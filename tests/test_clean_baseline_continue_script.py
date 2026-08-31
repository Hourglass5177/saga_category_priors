from __future__ import annotations

import re
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "category_priors"
    / "clean_baseline"
    / "continue_clean_baseline.sh"
)


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_continue_script_calls_only_registered_clean_experiment_entrypoint() -> None:
    source = _source()
    assert (
        '"$PYTHON_BIN" -m category_priors.clean_baseline.experiment '
        '--config "$CONFIG_PATH"'
    ) in source
    forbidden = (
        r"\b(?:wget|curl)\b",
        r"\bpip\s+install\b",
        r"\bconda\s+install\b",
        r"\bhdbscan\b",
        r"\bobjectbank\b",
        r"\bpostprocess(?:\.py)?\b",
        r"\bdownload\b",
        r"\btrain(?:ing)?\b",
    )
    for pattern in forbidden:
        assert re.search(pattern, source, flags=re.IGNORECASE) is None


def test_continue_script_audits_registered_resources_without_free() -> None:
    source = _source()
    assert "nvidia-smi" in source
    assert 'df -Pk "$DISK_PROBE"' in source
    assert "memory.current" in source
    assert "memory.max" in source
    assert "memory.events" in source
    assert "MIN_AVAILABLE_KIB=$((80 * 1024 * 1024))" in source
    assert re.search(r"\bfree\b", source) is None


def test_continue_script_uses_pid_claim_not_a_lock_file() -> None:
    source = _source()
    assert 'PID_PATH="$ARTIFACT_ROOT/clean_baseline_launcher.pid"' in source
    assert "set -o noclobber" in source
    assert "kill -0" in source
    assert "/proc/$candidate/cmdline" in source
    assert "flock" not in source
    assert ".lock" not in source


def test_continue_script_has_atomic_status_and_quoted_paths() -> None:
    source = _source()
    assert "os.replace(temporary, target)" in source
    assert 'mkdir -p "$ARTIFACT_ROOT"' in source
    assert 'cd "$REPO_ROOT"' in source
    assert '--config "$CONFIG_PATH"' in source
    assert '>> "$LOG_PATH" 2>&1' in source
    assert 'CONFIG_PATH=$1' in source


def test_continue_script_distinguishes_registered_stop_from_completion() -> None:
    source = _source()
    assert '"$ARTIFACT_ROOT/clean_baseline_status.json"' in source
    assert '"$EXPERIMENT_STATE" == "complete"' in source
    assert '"$EXPERIMENT_STATE" == "stopped"' in source
    assert "write_status stopped 0" in source
    assert "success without a terminal state" in source
