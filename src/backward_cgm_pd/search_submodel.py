"""Backward elimination on the model-inclusion lattice (Section 3.4)."""

from __future__ import annotations

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
from .rcon import ModelTester
from .search_tau import SearchResult, _test_many


def meet_submodel(first: ColoredGraph, second: ColoredGraph) -> ColoredGraph:
    """Meet for the model-inclusion order.

    The committed R function effectively intersects the three components.  Its
    two additional single-edge corrections index ``intersectMat(...)$ind``
    (a non-existent field), so they are no-ops in the code that produced the
    archived experiment results.  We preserve that observed behaviour here.
    """
    if first.p != second.p:
        raise ValueError("graphs must have the same p")
    return ColoredGraph(
        first.p,
        first.L_atomic & second.L_atomic,
        first.E & second.E,
        first.E_atomic & second.E_atomic,
    )


def saturated_neighbours(graph: ColoredGraph) -> list[ColoredGraph]:
    """Port of ``nei``: the covers below a saturated model."""
    output: list[ColoredGraph] = []
    p = graph.p
    for vertex in range(1, p // 2 + 1):
        output.append(
            ColoredGraph(p, graph.L_atomic - {vertex}, graph.E, graph.E_atomic)
        )
    for edge in sorted(full_edges(p).FT):
        output.append(
            ColoredGraph(p, graph.L_atomic, graph.E - {edge}, graph.E_atomic)
        )
    for edge in sorted(full_edges(p).FL):
        output.append(
            ColoredGraph(p, graph.L_atomic, graph.E, graph.E_atomic - {edge})
        )
        for removed in (edge, tau_edge(edge, p)):
            output.append(
                ColoredGraph(
                    p,
                    graph.L_atomic,
                    graph.E - {removed},
                    graph.E_atomic - {edge},
                )
            )
    return output


def backward_submodel(
    data: ArrayLike,
    *,
    itmax: int = 500,
    alpha: float = 0.05,
    optimizer_maxiter: int = 2_000,
    tolerance: float = 1e-9,
    n_jobs: int = 1,
    rcon_backend: str = "mle",
    verbose: bool = False,
) -> SearchResult:
    tester = ModelTester(
        data,
        maxiter=optimizer_maxiter,
        tolerance=tolerance,
        backend=rcon_backend,
        n_jobs=n_jobs,
    )
    saturated = tester.saturated.graph
    p = saturated.p
    fl = sorted(full_edges(p).FL)
    first_layer = saturated_neighbours(saturated)
    _, first_pvalues = _test_many(tester, first_layer)
    active_first = first_pvalues >= alpha

    # Layout of S1 is q vertex covers, q transverse covers, then triples per FL.
    triple_offset = p
    eligible_symmetric: list[Edge] = []
    for edge_index, edge in enumerate(fl):
        block = active_first[
            triple_offset + 3 * edge_index : triple_offset + 3 * edge_index + 3
        ]
        if len(block) == 3 and bool(np.all(block)):
            eligible_symmetric.append(edge)

    second_layer: list[ColoredGraph] = []
    second_edges: list[Edge] = []
    for edge in eligible_symmetric:
        model = ColoredGraph(
            p,
            saturated.L_atomic,
            saturated.E - {edge, tau_edge(edge, p)},
            saturated.E_atomic - {edge},
        )
        _, pvalue = tester.test(model)
        if pvalue >= alpha:
            second_layer.append(model)
            second_edges.append(edge)

    best = float(np.max(first_pvalues)) if len(first_pvalues) else -np.inf
    if best < alpha:
        return SearchResult(
            saturated, 0, 0, tester.saturated, 1.0, (saturated,)
        )
    selected_first: set[int] = {int(np.argmax(first_pvalues))}
    current = first_layer[next(iter(selected_first))]
    path = [saturated, current]
    iterations = 0
    number_models = 0

    while iterations < itmax:
        parts = out_edges(current.E, p)
        symmetric = parts.TE - current.E_atomic
        singles = current.E - parts.TE - tau_edges(parts.TE, p) - parts.ET
        relevant = symmetric | (singles & full_edges(p).FL) | tau_edges(
            singles & full_edges(p).FR, p
        )

        blocked_first: set[int] = set()
        for edge in relevant:
            if edge in fl:
                edge_index = fl.index(edge)
                blocked_first.update(
                    range(
                        triple_offset + 3 * edge_index,
                        triple_offset + 3 * edge_index + 3,
                    )
                )
        indices1 = [
            index
            for index in range(len(first_layer))
            if active_first[index]
            and index not in selected_first
            and index not in blocked_first
        ]
        neighbours1 = [
            meet_submodel(current, first_layer[index]) for index in indices1
        ]
        if neighbours1:
            _, pvalues1 = _test_many(tester, neighbours1)
            for index, pvalue in zip(indices1, pvalues1, strict=True):
                if pvalue < alpha:
                    active_first[index] = False
            local1 = int(np.argmax(pvalues1))
            candidate1 = neighbours1[local1]
            candidate1_p = float(pvalues1[local1])
            candidate1_index = indices1[local1]
        else:
            candidate1 = None
            candidate1_p = -np.inf
            candidate1_index = None

        indices2 = [
            index for index, edge in enumerate(second_edges) if edge in relevant
        ]
        neighbours2 = [
            meet_submodel(current, second_layer[index]) for index in indices2
        ]
        if neighbours2:
            _, pvalues2 = _test_many(tester, neighbours2)
            keep_indices: list[int] = []
            for index, pvalue in zip(indices2, pvalues2, strict=True):
                if pvalue < alpha:
                    keep_indices.append(index)
            for index in reversed(keep_indices):
                del second_layer[index]
                del second_edges[index]
            valid = [
                (model, float(pvalue))
                for model, pvalue in zip(neighbours2, pvalues2, strict=True)
                if pvalue >= alpha
            ]
            if valid:
                candidate2, candidate2_p = max(valid, key=lambda item: item[1])
            else:
                candidate2, candidate2_p = None, -np.inf
        else:
            candidate2, candidate2_p = None, -np.inf

        # R code recomputes which second-layer models remain possible from
        # the three acceptance flags belonging to every FL edge.
        still_eligible = {
            edge
            for edge_index, edge in enumerate(fl)
            if bool(
                np.all(
                    active_first[
                        triple_offset
                        + 3 * edge_index : triple_offset
                        + 3 * edge_index
                        + 3
                    ]
                )
            )
        }
        for index in reversed(range(len(second_edges))):
            if second_edges[index] not in still_eligible:
                del second_edges[index]
                del second_layer[index]

        tested_this_iteration = len(indices1) + len(indices2)
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
        if candidate1_p > candidate2_p:
            assert candidate1 is not None and candidate1_index is not None
            current = candidate1
            selected_first.add(candidate1_index)
        else:
            assert candidate2 is not None
            current = candidate2
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
