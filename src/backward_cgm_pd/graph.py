"""Graph representation and set operations ported from supplementary_functions.R.

Vertices intentionally remain 1-based.  This makes Python results directly
comparable with the R objects stored by the original project.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Iterator, Sequence

Edge = tuple[int, int]


def normalize_edge(edge: Sequence[int]) -> Edge:
    i, j = int(edge[0]), int(edge[1])
    if i == j:
        raise ValueError("Self-loops are not valid edges")
    return (i, j) if i < j else (j, i)


def edge_set(edges: Iterable[Sequence[int]] | None) -> frozenset[Edge]:
    if edges is None:
        return frozenset()
    return frozenset(normalize_edge(edge) for edge in edges)


def validate_p(p: int) -> None:
    if p <= 0 or p % 2:
        raise ValueError(f"p must be a positive even integer, got {p}")


def tau(vertex: int, p: int) -> int:
    validate_p(p)
    q = p // 2
    if not 1 <= vertex <= p:
        raise ValueError(f"vertex {vertex} is outside 1..{p}")
    return vertex + q if vertex <= q else vertex - q


def tau_edge(edge: Sequence[int], p: int) -> Edge:
    i, j = normalize_edge(edge)
    return normalize_edge((tau(i, p), tau(j, p)))


def tau_edges(edges: Iterable[Sequence[int]], p: int) -> frozenset[Edge]:
    return frozenset(tau_edge(edge, p) for edge in edges)


@dataclass(frozen=True)
class FullEdges:
    FV: frozenset[Edge]
    FL: frozenset[Edge]
    FR: frozenset[Edge]
    FT: frozenset[Edge]


def full_edges(p: int) -> FullEdges:
    """Return F_V, F_L, F_R and F_T using the definitions in the R code."""
    validate_p(p)
    fv = frozenset(combinations(range(1, p + 1), 2))
    ft = frozenset(edge for edge in fv if edge[1] == tau(edge[0], p))
    fl = frozenset(
        (i, j)
        for i, j in fv
        if (i < tau(1, p) and j < tau(1, p)) or i < tau(j, p)
    )
    fr = frozenset(
        (i, j)
        for i, j in fv
        if (i >= tau(1, p) and j >= tau(1, p))
        or (i > tau(j, p) and tau(j, p) > 0)
    )
    return FullEdges(fv, fl, fr, ft)


@dataclass(frozen=True)
class EdgeParts:
    ET: frozenset[Edge]
    EL: frozenset[Edge]
    ER: frozenset[Edge]
    TE: frozenset[Edge]


def out_edges(edges: Iterable[Sequence[int]], p: int) -> EdgeParts:
    current = edge_set(edges)
    all_edges = full_edges(p)
    el = all_edges.FL & current
    er = all_edges.FR & current
    et = all_edges.FT & current
    te = el & tau_edges(er, p)
    return EdgeParts(ET=et, EL=el, ER=er, TE=te)


@dataclass(frozen=True)
class ColoredGraph:
    """R representation ``list(L.as, E, E.as)``.

    ``L_atomic`` corresponds to ``L.as`` and ``E_atomic`` to ``E.as``.
    A twin pair not present in an atomic set shares one colour/parameter.
    """

    p: int
    L_atomic: frozenset[int]
    E: frozenset[Edge]
    E_atomic: frozenset[Edge]

    def __post_init__(self) -> None:
        validate_p(self.p)
        q = self.p // 2
        object.__setattr__(self, "L_atomic", frozenset(int(x) for x in self.L_atomic))
        object.__setattr__(self, "E", edge_set(self.E))
        object.__setattr__(self, "E_atomic", edge_set(self.E_atomic))
        if not self.L_atomic <= frozenset(range(1, q + 1)):
            raise ValueError("L_atomic must be a subset of the left vertices 1..p/2")
        if not self.E <= full_edges(self.p).FV:
            raise ValueError("E contains an invalid edge")
        if not self.E_atomic <= out_edges(self.E, self.p).TE:
            raise ValueError("E_atomic must contain left representatives of twin edges")

    @classmethod
    def saturated(cls, p: int) -> "ColoredGraph":
        edges = full_edges(p)
        return cls(
            p=p,
            L_atomic=frozenset(range(1, p // 2 + 1)),
            E=edges.FV,
            E_atomic=edges.FL,
        )

    @classmethod
    def from_r(
        cls,
        p: int,
        L_as: Iterable[int] | None,
        E: Iterable[Sequence[int]] | None,
        E_as: Iterable[Sequence[int]] | None,
    ) -> "ColoredGraph":
        return cls(p, frozenset(L_as or ()), edge_set(E), edge_set(E_as))

    def to_dict(self) -> dict[str, object]:
        return {
            "p": self.p,
            "L.as": sorted(self.L_atomic),
            "E": [list(edge) for edge in sorted(self.E)],
            "E.as": [list(edge) for edge in sorted(self.E_atomic)],
        }

    def vertex_colour_classes(self) -> tuple[frozenset[int], ...]:
        classes: list[frozenset[int]] = []
        for left in range(1, self.p // 2 + 1):
            right = tau(left, self.p)
            if left in self.L_atomic:
                classes.extend((frozenset((left,)), frozenset((right,))))
            else:
                classes.append(frozenset((left, right)))
        return tuple(classes)

    def edge_colour_classes(self) -> tuple[frozenset[Edge], ...]:
        twin_left = out_edges(self.E, self.p).TE
        symmetric = twin_left - self.E_atomic
        symmetric_full = symmetric | tau_edges(symmetric, self.p)
        classes: list[frozenset[Edge]] = [
            frozenset((edge, tau_edge(edge, self.p))) for edge in sorted(symmetric)
        ]
        classes.extend(frozenset((edge,)) for edge in sorted(self.E - symmetric_full))
        return tuple(classes)

    @property
    def n_parameters(self) -> int:
        return len(self.vertex_colour_classes()) + len(self.edge_colour_classes())


def intersect_edges(a: Iterable[Sequence[int]], b: Iterable[Sequence[int]]) -> frozenset[Edge]:
    return edge_set(a) & edge_set(b)


def difference_edges(a: Iterable[Sequence[int]], b: Iterable[Sequence[int]]) -> frozenset[Edge]:
    return edge_set(a) - edge_set(b)


def graph_key(graph: ColoredGraph) -> tuple[object, ...]:
    return (
        tuple(sorted(graph.L_atomic)),
        tuple(sorted(graph.E)),
        tuple(sorted(graph.E_atomic)),
    )


def unique_graphs(graphs: Iterable[ColoredGraph]) -> list[ColoredGraph]:
    seen: set[tuple[object, ...]] = set()
    result: list[ColoredGraph] = []
    for graph in graphs:
        key = graph_key(graph)
        if key not in seen:
            seen.add(key)
            result.append(graph)
    return result


def iter_edge_matrix(edges: Iterable[Edge]) -> Iterator[tuple[int, int]]:
    yield from sorted(edge_set(edges))
