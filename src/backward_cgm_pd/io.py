"""Interoperability with the original project's RData and JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rdata

from .graph import ColoredGraph


def read_rdata(path: str | Path) -> dict[str, Any]:
    return {str(key): value for key, value in rdata.read_rda(Path(path)).items()}


def load_simulated_datasets(path: str | Path) -> list[np.ndarray]:
    objects = read_rdata(path)
    if len(objects) != 1:
        raise ValueError(f"Expected one object in {path}, found {list(objects)}")
    datasets = next(iter(objects.values()))
    return [np.asarray(dataset, dtype=float) for dataset in datasets]


def _normalise_r_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected an R list converted to dict, got {type(value)!r}")
    return {str(key): item for key, item in value.items()}


def graph_from_r_mapping(value: Any, *, p: int | None = None) -> ColoredGraph:
    mapping = _normalise_r_mapping(value)
    if "model" in mapping:
        mapping = _normalise_r_mapping(mapping["model"])
    raw_edges = mapping.get("E")
    raw_atomic = mapping.get("E.as")
    raw_left = mapping.get("L.as")
    edges = np.asarray(
        np.empty((0, 2)) if raw_edges is None else raw_edges, dtype=int
    ).reshape(-1, 2)
    atomic_edges = np.asarray(
        np.empty((0, 2)) if raw_atomic is None else raw_atomic, dtype=int
    ).reshape(-1, 2)
    left = np.asarray(
        np.empty(0) if raw_left is None else raw_left, dtype=int
    ).reshape(-1)
    if p is None:
        candidates = [left.max(initial=0), edges.max(initial=0), atomic_edges.max(initial=0)]
        p = int(max(candidates))
    return ColoredGraph.from_r(p, left.tolist(), edges.tolist(), atomic_edges.tolist())


def load_saved_search(path: str | Path, *, p: int | None = None) -> tuple[ColoredGraph, dict[str, Any]]:
    objects = read_rdata(path)
    if len(objects) != 1:
        raise ValueError(f"Expected one object in {path}, found {list(objects)}")
    result = _normalise_r_mapping(next(iter(objects.values())))
    graph = graph_from_r_mapping(result, p=p)
    metadata = {
        key: value
        for key, value in result.items()
        if key not in {"model", "L.as", "E", "E.as"}
    }
    return graph, metadata


def load_saved_pdglasso(
    path: str | Path,
) -> tuple[ColoredGraph, dict[str, Any]]:
    """Load a saved pdRCON.fit object and apply pdColG.get defaults."""
    from .pdglasso import PDGlassoFit, graph_from_pdglasso

    objects = read_rdata(path)
    if len(objects) != 1:
        raise ValueError(f"Expected one object in {path}, found {list(objects)}")
    result = _normalise_r_mapping(next(iter(objects.values())))
    model = _normalise_r_mapping(result["model"])
    internal = _normalise_r_mapping(model["internal.par"])

    def scalar(key: str, default: float = 0.0) -> float:
        value = internal.get(key)
        if value is None:
            return default
        try:
            return float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError):
            return default

    fit = PDGlassoFit(
        precision=np.asarray(model["X"], dtype=float),
        lambda1=scalar("lambda1"),
        lambda2=scalar("lambda2"),
        converged=bool(scalar("converged")),
        iterations=int(scalar("n.iter")),
        primal_residual=scalar("res.primal"),
        dual_residual=scalar("res.dual"),
        execution_time=scalar("execution.time"),
        eps_relative=scalar("eps.rel", 1e-8),
    )
    graph = graph_from_pdglasso(fit)
    metadata = {
        "best_lambdas": np.asarray(
            result.get("best.lambdas", (fit.lambda1, fit.lambda2)),
            dtype=float,
        ).tolist(),
        "iterations": fit.iterations,
        "converged": fit.converged,
        "precision": fit.precision,
    }
    return graph, metadata


def load_air_quality_residuals(path: str | Path) -> np.ndarray:
    objects = read_rdata(path)
    if "data.res" not in objects:
        raise KeyError(f"data.res is not present in {path}")
    return np.asarray(objects["data.res"], dtype=float)


def write_json(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")

    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, ColoredGraph):
            return item.to_dict()
        raise TypeError(f"{type(item).__name__} is not JSON serialisable")

    temporary.write_text(
        json.dumps(value, indent=2, default=default), encoding="utf-8"
    )
    temporary.replace(destination)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
