from __future__ import annotations

"""Materialize the one mechanical CLI repair used by the 10k control."""

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

ITERATIONS_DEFAULT_OLD = "self.iterations = None"
ITERATIONS_DEFAULT_NEW = "self.iterations = 0"


def repair_iterations_cli(source: str) -> str:
    """Keep adaptive semantics while making ``--iterations`` an integer option."""

    count = source.count(ITERATIONS_DEFAULT_OLD)
    if count != 1:
        raise ValueError(f"expected one iterations default sentinel, found {count}")
    return source.replace(ITERATIONS_DEFAULT_OLD, ITERATIONS_DEFAULT_NEW, 1)


def materialize_iterations_cli_variant(
    full950_root: Path,
    destination: Path,
) -> dict[str, object]:
    """Copy exact full950 and change only the falsey iterations default to zero."""

    full950_root = full950_root.resolve()
    destination = destination.resolve()
    arguments_path = full950_root / "arguments/__init__.py"
    if not arguments_path.is_file():
        raise FileNotFoundError(arguments_path)
    if destination.exists():
        raise FileExistsError(destination)

    repaired = repair_iterations_cli(arguments_path.read_text(encoding="utf-8"))
    shutil.copytree(full950_root, destination)
    (destination / "arguments/__init__.py").write_text(repaired, encoding="utf-8")
    provenance = {
        "schema": "saga-baseline-iterations-cli-variant-v1",
        "source_commit": "95073c640a77984c6af24abb276147e4315abcd1",
        "source_root": str(full950_root),
        "destination": str(destination),
        "variant": "full950-iterations-cli",
        "repair": "OptimizationParams.iterations None -> 0",
        "adaptive_semantics_preserved": True,
    }
    (destination / "BASELINE_VARIANT.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full950-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = materialize_iterations_cli_variant(args.full950_root, args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
