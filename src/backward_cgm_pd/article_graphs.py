"""True colored graphs used to generate the eight saved simulation datasets."""

from __future__ import annotations

from dataclasses import dataclass

from .graph import ColoredGraph, Edge, full_edges, tau_edges
from .r_random import RRandom


@dataclass(frozen=True)
class ArticleGraphSpec:
    pair_seed: int
    pair_count: int
    atomic_count: int
    atomic_seed: int | None
    transverse_count: int
    include_transverse: bool
    rest_seed: int
    rest_count: int
    vertex_seed: int
    vertex_atomic_count: int
    atomic_take_first: bool = False


# Direct transcription of simulation/simulation.R.  The R file calls the
# transverse-edge component "FI", although fullEdges() returns it as "FT".
SPECS: dict[str, dict[int, ArticleGraphSpec]] = {
    "A": {
        8: ArticleGraphSpec(9, 2, 1, None, 0, False, 1, 1, 1, 3, True),
        # EI11.12 is constructed but not included in E11.12 in the R source.
        # The archived Table 3.2 symmetry results correspond to resetting
        # set.seed(2) before selecting E.as (as done explicitly for p=16/20).
        12: ArticleGraphSpec(2, 5, 4, 2, 1, False, 1, 2, 1, 4),
        16: ArticleGraphSpec(2, 9, 7, 2, 1, True, 1, 3, 1, 6),
        20: ArticleGraphSpec(2, 14, 11, 2, 2, True, 4, 4, 1, 8),
    },
    "B": {
        8: ArticleGraphSpec(4, 4, 1, 1, 0, False, 2, 2, 1, 1),
        12: ArticleGraphSpec(2, 9, 3, 2, 1, True, 3, 4, 1, 2),
        16: ArticleGraphSpec(2, 15, 5, 2, 3, True, 1, 9, 1, 2),
        20: ArticleGraphSpec(2, 24, 8, 2, 6, True, 1, 12, 1, 2),
    },
}


def _sample(items: list[Edge] | list[int], size: int, seed: int):
    return RRandom(seed).sample(items, size)


def article_scenario_graph(scenario: str, p: int) -> ColoredGraph:
    """Reconstruct ``g11.*``/``g22.*`` using R-compatible seeded sampling."""
    scenario = scenario.upper()
    try:
        spec = SPECS[scenario][p]
    except KeyError as error:
        raise ValueError(f"unsupported article scenario: {scenario}, p={p}") from error

    universe = full_edges(p)
    fl = sorted(universe.FL)
    ft = sorted(universe.FT)
    fv = sorted(universe.FV)

    pair_rng = RRandom(spec.pair_seed)
    pairs = pair_rng.sample(fl, spec.pair_count)
    if spec.atomic_take_first:
        atomic = pairs[: spec.atomic_count]
    else:
        atomic_rng = (
            pair_rng if spec.atomic_seed is None else RRandom(spec.atomic_seed)
        )
        atomic = atomic_rng.sample(pairs, spec.atomic_count)

    transverse = (
        set(_sample(ft, spec.transverse_count, 1))
        if spec.transverse_count
        else set()
    )
    paired_full = set(pairs) | set(tau_edges(pairs, p))
    rest_universe = [
        edge for edge in fv if edge not in paired_full and edge not in universe.FT
    ]
    extra = set(_sample(rest_universe, spec.rest_count, spec.rest_seed))
    edges = paired_full | extra
    if spec.include_transverse:
        edges |= transverse

    left_vertices = list(range(1, p // 2 + 1))
    vertex_atomic = _sample(
        left_vertices, spec.vertex_atomic_count, spec.vertex_seed
    )
    return ColoredGraph.from_r(p, vertex_atomic, edges, atomic)
