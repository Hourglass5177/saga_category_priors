from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import hash_json, load_json


@dataclass(frozen=True)
class Taxonomy:
    schema_version: str
    benchmark_name: str
    canonical_classes: tuple[str, ...]
    dataset_mappings: dict[str, dict[str, str]]
    parents: dict[str, str]
    unsupported_saga_classes: tuple[str, ...]
    content_hash: str

    def map_label(self, dataset: str, raw_label: str) -> str | None:
        mapping = self.dataset_mappings.get(dataset.lower())
        if mapping is None:
            raise KeyError(f"Unknown dataset mapping: {dataset}")
        return mapping.get(normalize_label(raw_label))

    def parent_for(self, canonical_class: str) -> str:
        return self.parents.get(canonical_class, "global")

    def validate(self) -> None:
        allowed = set(self.canonical_classes)
        if len(allowed) != len(self.canonical_classes):
            raise ValueError("canonical_classes contains duplicates")
        for dataset, mapping in self.dataset_mappings.items():
            if len(mapping) != len(set(mapping.values())):
                # Several raw names may intentionally map to one class in future schemas;
                # v1 forbids that so the primary protocol remains exact/synonym only.
                raise ValueError(f"{dataset}: v1 mapping must be one-to-one")
            unknown = set(mapping.values()) - allowed
            if unknown:
                raise ValueError(
                    f"{dataset}: unknown canonical classes: {sorted(unknown)}"
                )
        unknown_parents = set(self.parents) - allowed
        if unknown_parents:
            raise ValueError(
                f"parents contains unknown classes: {sorted(unknown_parents)}"
            )


def normalize_label(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def default_taxonomy_path() -> Path:
    return Path(__file__).with_name("default_taxonomy.json")


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    source = Path(path) if path is not None else default_taxonomy_path()
    payload: dict[str, Any] = load_json(source)
    mappings = {
        normalize_label(dataset): {
            normalize_label(raw): normalize_label(canonical)
            for raw, canonical in mapping.items()
        }
        for dataset, mapping in payload["dataset_mappings"].items()
    }
    taxonomy = Taxonomy(
        schema_version=str(payload["schema_version"]),
        benchmark_name=str(payload["benchmark_name"]),
        canonical_classes=tuple(
            normalize_label(item) for item in payload["canonical_classes"]
        ),
        dataset_mappings=mappings,
        parents={
            normalize_label(key): normalize_label(value)
            for key, value in payload["parents"].items()
        },
        unsupported_saga_classes=tuple(
            normalize_label(item)
            for item in payload.get("unsupported_saga_classes", [])
        ),
        content_hash=hash_json(payload),
    )
    taxonomy.validate()
    return taxonomy
