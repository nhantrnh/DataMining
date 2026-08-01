"""Recovery metrics used in Table 3.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .graph import ColoredGraph, full_edges, out_edges


@dataclass(frozen=True)
class RecoveryMetrics:
    ePPV: float
    eTPR: float
    eTNR: float
    sPPV: float
    sTPR: float
    sTNR: float

    def to_dict(self) -> dict[str, float]:
        return {
            "ePPV": self.ePPV,
            "eTPR": self.eTPR,
            "eTNR": self.eTNR,
            "sPPV": self.sPPV,
            "sTPR": self.sTPR,
            "sTNR": self.sTNR,
        }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def recovery_metrics(estimate: ColoredGraph, truth: ColoredGraph) -> RecoveryMetrics:
    if estimate.p != truth.p:
        raise ValueError("estimate and truth must have the same p")
    universe = full_edges(truth.p).FV
    edge_tp = len(estimate.E & truth.E)
    edge_tn = len((universe - estimate.E) & (universe - truth.E))

    estimate_pairs = out_edges(estimate.E, estimate.p).TE
    true_pairs = out_edges(truth.E, truth.p).TE
    estimate_symmetric = estimate_pairs - estimate.E_atomic
    true_symmetric = true_pairs - truth.E_atomic
    symmetry_tp = len(estimate_symmetric & true_symmetric)

    # Exact translation of simulation.R: sTNR is evaluated only on twin
    # edge-pairs formed by missing edges, not on every possible FL statement.
    estimate_missing_pairs = out_edges(universe - estimate.E, estimate.p).TE
    true_missing_pairs = out_edges(universe - truth.E, truth.p).TE
    symmetry_tn = len(estimate_missing_pairs & true_missing_pairs)
    return RecoveryMetrics(
        ePPV=_ratio(edge_tp, len(estimate.E)),
        eTPR=_ratio(edge_tp, len(truth.E)),
        eTNR=_ratio(edge_tn, len(universe - truth.E)),
        sPPV=_ratio(symmetry_tp, len(estimate_symmetric)),
        sTPR=_ratio(symmetry_tp, len(true_symmetric)),
        sTNR=_ratio(symmetry_tn, len(true_missing_pairs)),
    )


def average_metrics(metrics: Iterable[RecoveryMetrics]) -> dict[str, float]:
    rows = [row.to_dict() for row in metrics]
    if not rows:
        return {}
    return {
        # R's plain mean() propagates NA/NaN; it does not use na.rm=TRUE.
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
