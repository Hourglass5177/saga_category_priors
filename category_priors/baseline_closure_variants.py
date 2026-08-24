from __future__ import annotations

"""Materialize the two minimal repairs between bfc2192 and 95073c6."""

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

ARGS_SIGNATURE_OLD = (
    "def training(dataset, opt, pipe, iteration, saving_iterations, "
    "checkpoint_iterations, debug_from):"
)
ARGS_SIGNATURE_NEW = (
    "def training(args, dataset, opt, pipe, iteration, saving_iterations, "
    "checkpoint_iterations, debug_from):"
)
ARGS_CALL_OLD = (
    "training(lp.extract(args), op.extract(args), pp.extract(args), "
    "args.iteration, args.save_iterations, args.checkpoint_iterations, "
    "args.debug_from)"
)
ARGS_CALL_NEW = (
    "training(args, lp.extract(args), op.extract(args), pp.extract(args), "
    "args.iteration, args.save_iterations, args.checkpoint_iterations, "
    "args.debug_from)"
)
NORMALIZE_OLD = (
    "torch.nn.functional.normalize(sample_features[None,...]*gates[:,None,...])"
)
NORMALIZE_NEW = (
    "torch.nn.functional.normalize(sample_features[None,...]*gates[:,None,...], dim=-1)"
)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one source sentinel, found {count}")
    return text.replace(old, new, 1)


def repair_training_source(source: str, variant: str) -> str:
    """Apply only registered mechanical repairs to a bfc2192 train script."""

    if variant not in {"args-only", "args-norm"}:
        raise ValueError(f"unsupported partial repair variant: {variant}")
    repaired = _replace_once(source, ARGS_SIGNATURE_OLD, ARGS_SIGNATURE_NEW, "args")
    repaired = _replace_once(repaired, ARGS_CALL_OLD, ARGS_CALL_NEW, "args-call")
    if variant == "args-norm":
        repaired = _replace_once(
            repaired, NORMALIZE_OLD, NORMALIZE_NEW, "feature-normalization"
        )
    return repaired


def materialize_training_variant(
    bfc_root: Path,
    destination: Path,
    variant: str,
) -> dict[str, object]:
    """Copy an exported bfc2192 tree and apply one registered partial repair."""

    bfc_root = bfc_root.resolve()
    destination = destination.resolve()
    source_script = bfc_root / "train_contrastive_feature.py"
    if not source_script.is_file():
        raise FileNotFoundError(source_script)
    if destination.exists():
        raise FileExistsError(destination)
    source = source_script.read_text(encoding="utf-8")
    repaired = repair_training_source(source, variant)

    shutil.copytree(bfc_root, destination)
    (destination / "train_contrastive_feature.py").write_text(
        repaired, encoding="utf-8"
    )
    repairs = ["pass_cli_args_into_training"]
    if variant == "args-norm":
        repairs.append("normalize_distance_regularizer_on_feature_channel")
    provenance = {
        "schema": "saga-baseline-partial-repair-v1",
        "source_commit": "bfc21922384cc991a71b5e51429354b5d6b06375",
        "source_root": str(bfc_root),
        "destination": str(destination),
        "variant": variant,
        "repairs": repairs,
    }
    (destination / "BASELINE_VARIANT.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfc-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for variant in ("args-only", "args-norm"):
        result = materialize_training_variant(
            args.bfc_root, args.output_root / variant, variant
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
