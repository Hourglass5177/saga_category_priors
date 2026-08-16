from __future__ import annotations

"""Read-only provenance report for the public SAGA fork chain used by V6."""

import subprocess
from pathlib import Path
from typing import Any

from .io import write_json


PUBLIC_REPOSITORY = "https://github.com/Jumpat/SegAnyGAussians.git"


def _git(repository: str | Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _stat(repository: str | Path, left: str, right: str) -> str | None:
    try:
        return _git(repository, "diff", "--stat", f"{left}..{right}")
    except RuntimeError:
        return None


def audit_v6_provenance(
    *, repository: str | Path, output: str | Path,
    public_repository: str = PUBLIC_REPOSITORY,
) -> dict[str, Any]:
    repo = Path(repository).resolve()
    head = _git(repo, "rev-parse", "HEAD")
    refs: dict[str, str | None] = {}
    for ref in ("source/v2", "source/a800"):
        try:
            refs[ref] = _git(repo, "rev-parse", ref)
        except RuntimeError:
            refs[ref] = None
    source_url = None
    try:
        source_url = _git(repo, "remote", "get-url", "source")
    except RuntimeError:
        pass
    public_head = None
    try:
        result = subprocess.run(
            ["git", "ls-remote", public_repository, "HEAD"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            public_head = result.stdout.split()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    payload = {
        "kind": "v6_provenance_audit", "repository": str(repo), "current_head": head,
        "source_remote": source_url, "public_repository": public_repository,
        "public_head_at_audit": public_head,
        "refs": refs,
        "diff_stats": {
            "source_v2_to_source_a800": _stat(repo, "source/v2", "source/a800") if refs["source/v2"] and refs["source/a800"] else None,
            "source_a800_to_current": _stat(repo, "source/a800", head) if refs["source/a800"] else None,
        },
        "conclusion": (
            "source/v2 is a local historical ref, not asserted byte-identical to the public repository; "
            "the public HEAD is recorded separately for transparent later comparison."
        ),
    }
    write_json(output, payload)
    return payload
