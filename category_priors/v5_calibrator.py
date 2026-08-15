from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_rows, write_json


FEATURES = ("E", "G", "C")


def _fit(rows: Sequence[Mapping[str, Any]], feature_prefix: str) -> dict[str, Any]:
    x = np.asarray(
        [[float(row[f"{feature_prefix}_{name}"]) for name in FEATURES] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([float(row["same_class_best_iou"]) >= 0.50 for row in rows], dtype=np.int64)
    if len(np.unique(y)) != 2:
        raise ValueError("V5 calibrator requires both positive and negative dev candidates")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-12)
    x = (x - mean) / scale
    # Fixed full-batch L2 logistic regression.  It avoids making the recovery
    # experiment depend on a particular sklearn version while retaining the
    # registered C=1, balanced-weight model and no class-ID feature.
    positive_weight = len(y) / (2.0 * float(y.sum()))
    negative_weight = len(y) / (2.0 * float((1 - y).sum()))
    weights = np.where(y == 1, positive_weight, negative_weight)
    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    intercept = 0.0
    for _ in range(2000):
        logits = np.clip(intercept + x @ coefficients, -50.0, 50.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        residual = weights * (probability - y)
        gradient = x.T @ residual / len(y) + coefficients / len(y)  # C=1
        intercept_gradient = float(residual.mean())
        coefficients -= 0.10 * gradient
        intercept -= 0.10 * intercept_gradient
    return {
        "feature_names": list(FEATURES), "coefficients": coefficients.astype(float).tolist(),
        "intercept": float(intercept), "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(), "candidate_count": int(len(rows)),
        "positive_count": int(y.sum()), "negative_count": int((1 - y).sum()),
    }


def fit_v5_calibrator(
    candidate_table: str | Path, output: str | Path, *, source: str,
    development_scenes: Sequence[str],
) -> dict[str, Any]:
    """Fit only fixed dev8 rows.  The output can be replayed without GT access."""
    allowed = {str(value) for value in development_scenes}
    rows = [
        row for row in read_rows(candidate_table)
        if row.get("row_type") == "candidate" and row.get("source") == source
        and str(row.get("scene_id")) in allowed
    ]
    if not rows:
        raise ValueError("no V5 development candidate rows found")
    payload = {
        "kind": "v5_logistic_calibrators", "schema_version": "1.0", "source": source,
        "development_scenes": sorted(allowed), "positive_definition": "same_class_gt_best_iou >= 0.50",
        "C": 1.0, "class_weight": "balanced", "class_id_feature": False,
        "threshold": 0.50,
        "uniform": _fit(rows, "uniform"), "class": _fit(rows, "class"),
    }
    write_json(output, payload)
    return payload
