"""Backward elimination on the twin lattice (Section 3.3 / Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from .graph import (
    ColoredGraph,
    Edge,
    full_edges,
    out_edges,
    tau_edge,
    tau_edges,
)
from .rcon import ModelTester, RCONFit


@dataclass(frozen=True)
class SearchResult:
    model: ColoredGraph
    iterations: int
    number_models: int
    fit: RCONFit
    pvalue: float
    path: tuple[ColoredGraph, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model.to_dict(),
            "iterations": self.iterations,
            "no.models": self.number_models,
            "pvalue": self.pvalue,
            "path": [model.to_dict() for model in self.path],
        }


def meet_twin(first: ColoredGraph, second: ColoredGraph) -> ColoredGraph:
    if first.p != second.p:
        raise ValueError("graphs must have the same p")
    return ColoredGraph(
        p=first.p,
        L_atomic=first.L_atomic & second.L_atomic,
        E=first.E & second.E,
        E_atomic=first.E_atomic & second.E_atomic,
    )


def upper_layer(graph: ColoredGraph) -> list[ColoredGraph]:
    """The R ``layer1`` construction, normally called for the saturated graph."""
    output: list[ColoredGraph] = []
    for vertex in sorted(graph.L_atomic):
        output.append(
            ColoredGraph(graph.p, graph.L_atomic - {vertex}, graph.E, graph.E_atomic)
        )
    for edge in sorted(out_edges(graph.E, graph.p).ET):
        output.append(
            ColoredGraph(graph.p, graph.L_atomic, graph.E - {edge}, graph.E_atomic)
        )
    for edge in sorted(graph.E_atomic):
        output.append(
            ColoredGraph(
                graph.p, graph.L_atomic, graph.E, graph.E_atomic - {edge}
            )
        )
    return output


def lower_layer(
    graph: ColoredGraph, rejected_atomic_edges: Iterable[Edge]
) -> list[ColoredGraph]:
    output: list[ColoredGraph] = []
    for edge in rejected_atomic_edges:
        for removed in (edge, tau_edge(edge, graph.p)):
            output.append(
                ColoredGraph(
                    graph.p,
                    graph.L_atomic,
                    graph.E - {removed},
                    graph.E_atomic - {edge},
                )
            )
    return output


def third_layer(
    graph: ColoredGraph, accepted_atomic_edges: Iterable[Edge]
) -> list[ColoredGraph]:
    return [
        ColoredGraph(
            graph.p,
            graph.L_atomic,
            graph.E - {edge, tau_edge(edge, graph.p)},
            graph.E_atomic - {edge},
        )
        for edge in accepted_atomic_edges
    ]


def _test_many(
    tester: ModelTester, models: list[ColoredGraph]
) -> tuple[list[RCONFit], np.ndarray]:
    tested = tester.test_many(models)
    return [item[0] for item in tested], np.asarray([item[1] for item in tested])


def backward_cgm_pd(
    data: ArrayLike,
    *,
    itmax: int = 500,
    alpha: float = 0.05,
    optimizer_maxiter: int = 2_000,
    tolerance: float = 1e-9,
    n_jobs: int = 1,
    rcon_backend: str = "mle",
    count_initial_models: bool = True,
    verbose: bool = False,
) -> SearchResult:
    """Faithful Python port of ``backwardCGMpd``.

    Candidate pruning follows the three layers used by the R implementation.
    ``number_models`` includes the three initial layers.  This matches the
    archived simulation output produced by the uncommitted
    ``backwardCGMpd1`` variant called from ``simulation.R``.
    """
    tester = ModelTester(
        data,
        maxiter=optimizer_maxiter,
        tolerance=tolerance,
        backend=rcon_backend,
        n_jobs=n_jobs,
    )
    saturated = tester.saturated.graph
    p = saturated.p

    layer1 = upper_layer(saturated)
    _, pvalues1_initial = _test_many(tester, layer1)
    accepted1 = pvalues1_initial >= alpha
    active1 = accepted1.copy()

    edge_offset = p  # q vertex + q transverse-edge removals
    atomic_edges = sorted(saturated.E_atomic)
    accepted_edge_positions = [
        index - edge_offset
        for index in np.flatnonzero(accepted1)
        if index >= edge_offset
    ]
    rejected_edge_positions = [
        index - edge_offset
        for index in np.flatnonzero(~accepted1)
        if index >= edge_offset
    ]
    accepted_edges = [atomic_edges[index] for index in accepted_edge_positions]
    rejected_edges = [atomic_edges[index] for index in rejected_edge_positions]

    layer2 = lower_layer(saturated, rejected_edges)
    if layer2:
        _, pvalues2_initial = _test_many(tester, layer2)
        active2 = pvalues2_initial >= alpha
    else:
        pvalues2_initial = np.empty(0)
        active2 = np.empty(0, dtype=bool)

    layer3 = third_layer(saturated, accepted_edges)
    if layer3:
        _, pvalues3_initial = _test_many(tester, layer3)
        active3 = pvalues3_initial >= alpha
    else:
        pvalues3_initial = np.empty(0)
        active3 = np.empty(0, dtype=bool)

    best1 = float(np.max(pvalues1_initial)) if len(pvalues1_initial) else -np.inf
    best2 = float(np.max(pvalues2_initial)) if len(pvalues2_initial) else -np.inf
    if max(best1, best2) < alpha:
        return SearchResult(
            saturated,
            0,
            0,
            tester.saturated,
            1.0,
            (saturated,),
        )

    selected1: set[int] = set()
    selected2: set[int] = set()
    selected2_twins: set[int] = set()
    selected3: set[int] = set()
    if best1 >= best2:
        chosen = int(np.argmax(pvalues1_initial))
        selected1.add(chosen)
        current = layer1[chosen]
    else:
        chosen = int(np.argmax(pvalues2_initial))
        selected2.add(chosen)
        selected2_twins.add(chosen - 1 if chosen % 2 else chosen + 1)
        current = layer2[chosen]

    path = [saturated, current]
    iterations = 0
    number_models = (
        len(layer1) + len(layer2) + len(layer3)
        if count_initial_models
        else 0
    )
    while iterations < itmax:
        indices1 = [
            index
            for index in range(len(layer1))
            if active1[index] and index not in selected1
        ]
        neighbours1 = [meet_twin(current, layer1[index]) for index in indices1]
        if neighbours1:
            _, pvalues1 = _test_many(tester, neighbours1)
            for index, pvalue in zip(indices1, pvalues1, strict=True):
                if pvalue < alpha:
                    active1[index] = False
            local1 = int(np.argmax(pvalues1))
            candidate1 = neighbours1[local1]
            candidate1_p = float(pvalues1[local1])
            candidate1_source = ("layer1", indices1[local1])
        else:
            candidate1 = None
            candidate1_p = -np.inf
            candidate1_source = None

        symmetric_current = out_edges(current.E, p).TE - current.E_atomic
        indices3 = [
            index
            for index, edge in enumerate(accepted_edges)
            if edge in symmetric_current
            and active3[index]
            and index not in selected3
        ]
        neighbours3 = [meet_twin(current, layer3[index]) for index in indices3]
        if neighbours3:
            _, pvalues3 = _test_many(tester, neighbours3)
            for index, pvalue in zip(indices3, pvalues3, strict=True):
                if pvalue < alpha:
                    active3[index] = False
            local3 = int(np.argmax(pvalues3))
            if float(pvalues3[local3]) >= candidate1_p:
                candidate1 = neighbours3[local3]
                candidate1_p = float(pvalues3[local3])
                candidate1_source = ("layer3", indices3[local3])

        excluded2 = selected2 | selected2_twins
        indices2 = [
            index
            for index in range(len(layer2))
            if active2[index] and index not in excluded2
        ]
        neighbours2 = [meet_twin(current, layer2[index]) for index in indices2]
        if neighbours2:
            _, pvalues2 = _test_many(tester, neighbours2)
            for index, pvalue in zip(indices2, pvalues2, strict=True):
                if pvalue < alpha:
                    active2[index] = False
            local2 = int(np.argmax(pvalues2))
            candidate2 = neighbours2[local2]
            candidate2_p = float(pvalues2[local2])
            candidate2_index = indices2[local2]
        else:
            candidate2 = None
            candidate2_p = -np.inf
            candidate2_index = None

        tested_this_iteration = len(indices1) + len(indices3) + len(indices2)
        if tested_this_iteration:
            iterations += 1
        number_models += tested_this_iteration
        if verbose:
            print(
                f"iteration={iterations} candidates={tested_this_iteration} "
                f"best_p={max(candidate1_p, candidate2_p):.6g}"
            )

        if max(candidate1_p, candidate2_p) < alpha:
            break
        if candidate1_p >= candidate2_p:
            assert candidate1 is not None and candidate1_source is not None
            current = candidate1
            source, index = candidate1_source
            (selected1 if source == "layer1" else selected3).add(index)
        else:
            assert candidate2 is not None and candidate2_index is not None
            current = candidate2
            selected2.add(candidate2_index)
            selected2_twins.add(
                candidate2_index - 1
                if candidate2_index % 2
                else candidate2_index + 1
            )
        path.append(current)

    final_fit, final_pvalue = tester.test(current)
    return SearchResult(
        current,
        iterations,
        number_models,
        final_fit,
        final_pvalue,
        tuple(path),
    )
