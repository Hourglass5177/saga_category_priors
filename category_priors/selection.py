from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .io import hash_json
from .scannet import physical_scene_id
from .taxonomy import Taxonomy


def _scene_class_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if str(row.get("split", "")).lower() not in {"val", "validation"}:
            raise ValueError("Scene selection accepts validation rows only")
        counts[str(row["scene_id"])][str(row["canonical_class"])] += 1
    return {scene: dict(values) for scene, values in counts.items()}


def _partition_physical_scenes(
    counts: Mapping[str, Mapping[str, int]],
    canonical_classes: Sequence[str],
) -> tuple[set[str], set[str]]:
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for scene_id, class_counts in counts.items():
        group = physical_scene_id(scene_id)
        for category, count in class_counts.items():
            grouped[group][category] += int(count)

    groups = sorted(grouped)
    global_totals = {
        category: sum(grouped[group].get(category, 0) for group in groups)
        for category in canonical_classes
    }
    target_tune = {category: value / 3.0 for category, value in global_totals.items()}
    tune_totals = defaultdict(int)
    tune: set[str] = set()
    lock: set[str] = set()

    def rarity_score(group: str) -> float:
        return sum(
            grouped[group].get(category, 0) / max(global_totals[category], 1)
            for category in canonical_classes
        )

    for group in sorted(groups, key=lambda item: (-rarity_score(item), item)):
        desired_group_count = len(groups) / 3.0
        tune_cost = sum(
            abs(
                (tune_totals[category] + grouped[group].get(category, 0))
                - target_tune[category]
            )
            / max(global_totals[category], 1)
            for category in canonical_classes
        ) + 0.25 * abs((len(tune) + 1) - desired_group_count) / max(len(groups), 1)
        lock_cost = sum(
            abs(tune_totals[category] - target_tune[category])
            / max(global_totals[category], 1)
            for category in canonical_classes
        ) + 0.25 * abs(len(tune) - desired_group_count) / max(len(groups), 1)
        if tune_cost < lock_cost or (
            np.isclose(tune_cost, lock_cost) and len(tune) < desired_group_count
        ):
            tune.add(group)
            for category in canonical_classes:
                tune_totals[category] += grouped[group].get(category, 0)
        else:
            lock.add(group)
    return tune, lock


def _greedy_select(
    candidates: Sequence[str],
    counts: Mapping[str, Mapping[str, int]],
    canonical_classes: Sequence[str],
    scene_budget: int,
    target_per_class: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    available_totals = {
        category: sum(
            int(counts.get(scene, {}).get(category, 0)) for scene in candidates
        )
        for category in canonical_classes
    }
    selected: list[str] = []
    covered = defaultdict(int)
    remaining = set(candidates)
    while remaining and len(selected) < scene_budget:
        best_scene: str | None = None
        best_gain = -1.0
        for scene in sorted(remaining):
            gain = 0.0
            for category in canonical_classes:
                count = int(counts.get(scene, {}).get(category, 0))
                if count <= 0:
                    continue
                need = max(target_per_class - covered[category], 0)
                rarity = 1.0 / max(available_totals[category], 1)
                gain += min(count, need) * (1.0 + 10.0 * rarity)
                if covered[category] == 0:
                    gain += 3.0
            if gain > best_gain or (
                np.isclose(gain, best_gain)
                and (best_scene is None or scene < best_scene)
            ):
                best_scene = scene
                best_gain = gain
        assert best_scene is not None
        selected.append(best_scene)
        remaining.remove(best_scene)
        for category, count in counts.get(best_scene, {}).items():
            covered[category] += int(count)

    replacements = sorted(
        remaining,
        key=lambda scene: (
            -sum(
                min(int(counts.get(scene, {}).get(category, 0)), target_per_class)
                for category in canonical_classes
            ),
            scene,
        ),
    )
    return selected, replacements, dict(covered)


def _greedy_select_independent_scenes(
    candidates: Sequence[str],
    counts: Mapping[str, Mapping[str, int]],
    canonical_classes: Sequence[str],
    scene_budget: int,
    target_per_class: int,
) -> tuple[list[str], dict[str, int]]:
    """Select at most one scan from each physical scene."""
    available_totals = {
        category: sum(
            int(counts.get(scene, {}).get(category, 0)) for scene in candidates
        )
        for category in canonical_classes
    }
    selected: list[str] = []
    covered = defaultdict(int)
    remaining = set(candidates)
    while remaining and len(selected) < scene_budget:
        best_scene: str | None = None
        best_gain = -1.0
        for scene in sorted(remaining):
            gain = 0.0
            for category in canonical_classes:
                count = int(counts.get(scene, {}).get(category, 0))
                if count <= 0:
                    continue
                need = max(target_per_class - covered[category], 0)
                rarity = 1.0 / max(available_totals[category], 1)
                gain += min(count, need) * (1.0 + 10.0 * rarity)
                if covered[category] == 0:
                    gain += 3.0
            if gain > best_gain or (
                np.isclose(gain, best_gain)
                and (best_scene is None or scene < best_scene)
            ):
                best_scene = scene
                best_gain = gain
        assert best_scene is not None
        selected.append(best_scene)
        selected_group = physical_scene_id(best_scene)
        remaining = {
            scene
            for scene in remaining
            if physical_scene_id(scene) != selected_group
        }
        for category, count in counts.get(best_scene, {}).items():
            covered[category] += int(count)

    return selected, dict(covered)


def select_locked_evaluation_scenes(
    rows: Sequence[Mapping[str, Any]],
    taxonomy: Taxonomy,
    previous_selection: Mapping[str, Any],
    locked_budget: int = 48,
    target_per_class: int = 20,
) -> dict[str, Any]:
    """Choose independent locked scans from a previously registered split."""
    counts = _scene_class_counts(rows)
    selection = previous_selection["selection"]
    candidates = list(
        dict.fromkeys(
            str(scene)
            for key in ("locked", "locked_replacements")
            for scene in selection[key]
        )
    )
    tune_scenes = [
        str(scene)
        for key in ("tune", "tune_replacements")
        for scene in selection[key]
    ]
    candidate_groups = {physical_scene_id(scene) for scene in candidates}
    tune_groups = {physical_scene_id(scene) for scene in tune_scenes}
    if candidate_groups & tune_groups:
        raise ValueError(
            "Physical scene leakage between tune pool and locked candidate pool"
        )
    if len(candidate_groups) < locked_budget:
        raise ValueError(
            "Insufficient independent locked candidates: "
            f"groups={len(candidate_groups)}, required={locked_budget}"
        )
    missing = [scene for scene in candidates if scene not in counts]
    if missing:
        raise ValueError(f"Locked candidates missing validation statistics: {missing}")

    selected, coverage = _greedy_select_independent_scenes(
        candidates,
        counts,
        taxonomy.canonical_classes,
        locked_budget,
        target_per_class,
    )
    return {
        "schema_version": "1.0",
        "kind": "locked_evaluation_scenes",
        "benchmark_name": taxonomy.benchmark_name,
        "split": "val-locked",
        "scenes": [
            {
                "scene_id": scene,
                "physical_scene_id": physical_scene_id(scene),
            }
            for scene in selected
        ],
        "coverage": {
            category: coverage.get(category, 0)
            for category in taxonomy.canonical_classes
        },
        "target_per_class": target_per_class,
        "candidate_scan_count": len(candidates),
        "candidate_physical_scene_count": len(candidate_groups),
    }


def select_scenes(
    rows: Sequence[Mapping[str, Any]],
    taxonomy: Taxonomy,
    tune_budget: int = 24,
    locked_budget: int = 48,
    tune_target_per_class: int = 10,
    locked_target_per_class: int = 20,
    seed: int = 20260804,
) -> dict[str, Any]:
    # The algorithm is deterministic; seed is retained in the manifest so any future
    # randomized implementation cannot silently change the registered split.
    counts = _scene_class_counts(rows)
    tune_groups, locked_groups = _partition_physical_scenes(
        counts, taxonomy.canonical_classes
    )
    tune_candidates = sorted(
        scene for scene in counts if physical_scene_id(scene) in tune_groups
    )
    locked_candidates = sorted(
        scene for scene in counts if physical_scene_id(scene) in locked_groups
    )
    if len(tune_candidates) < tune_budget or len(locked_candidates) < locked_budget:
        raise ValueError(
            f"Insufficient validation scenes after grouped partition: tune={len(tune_candidates)}, "
            f"locked={len(locked_candidates)}"
        )
    tune, tune_replacements, tune_coverage = _greedy_select(
        tune_candidates,
        counts,
        taxonomy.canonical_classes,
        tune_budget,
        tune_target_per_class,
    )
    locked, locked_replacements, locked_coverage = _greedy_select(
        locked_candidates,
        counts,
        taxonomy.canonical_classes,
        locked_budget,
        locked_target_per_class,
    )
    if {physical_scene_id(scene) for scene in tune} & {
        physical_scene_id(scene) for scene in locked
    }:
        raise AssertionError("Physical scene leakage between tune and locked sets")
    payload = {
        "schema_version": "1.0",
        "kind": "scene_selection",
        "benchmark_name": taxonomy.benchmark_name,
        "taxonomy_sha256": taxonomy.content_hash,
        "seed": seed,
        "selection": {
            "tune": tune,
            "locked": locked,
            "tune_replacements": tune_replacements,
            "locked_replacements": locked_replacements,
        },
        "coverage": {
            "tune": {
                category: tune_coverage.get(category, 0)
                for category in taxonomy.canonical_classes
            },
            "locked": {
                category: locked_coverage.get(category, 0)
                for category in taxonomy.canonical_classes
            },
        },
        "targets": {
            "tune_per_class": tune_target_per_class,
            "locked_per_class": locked_target_per_class,
        },
    }
    payload["content_sha256"] = hash_json(payload)
    return payload
