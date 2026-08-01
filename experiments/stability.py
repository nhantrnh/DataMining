"""Bootstrap stability of the model selected on the twin lattice.

Motivation
----------
Roverato & Nguyen (2024) report, for every simulated configuration, the
average recovery of a *single* model per replicate.  Averages of that kind say
how often the procedure is right on a fresh sample, but they say nothing about
how sensitive one particular fitted model is to the sample that produced it.
Two configurations with identical average recovery can behave very
differently: one may return nearly the same graph under resampling, the other
a different graph almost every time.

This script quantifies that missing dimension with a nonparametric bootstrap
in the spirit of stability selection (Meinshausen & Bühlmann, 2010).  For a
given dataset it

1. runs the coherent backward elimination once to obtain the point estimate,
2. draws ``B`` bootstrap resamples of the *rows* and re-runs the search on
   each one,
3. reports, for every edge and for every twin-symmetry statement, the fraction
   of resamples in which it is selected.

Only rows are resampled.  The twin structure of the model is positional -
column ``j`` is paired with column ``j + p/2`` - so a row already carries a
complete pair and row resampling is the correct nonparametric unit.  Permuting
or subsetting columns would destroy the pairing the method is built on.

Two derived quantities summarise a run:

``instability``
    mean over all vertex pairs of ``2 f (1 - f)``, where ``f`` is the
    selection frequency of the corresponding edge.  It is zero when every
    resample returns the same edge set and reaches its maximum of 0.5 when
    edges are selected by a coin flip.  This is the criterion of Sun, Wang &
    Fang (2013) applied to the edge set.

``exact_recovery_rate``
    fraction of resamples returning exactly the point estimate, colour classes
    included.

Usage::

    python experiments/stability.py --scenario A --p 8 --replicate 1 \
        --bootstrap 200 --output results/stability-A-p8.json
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from backward_cgm_pd.article_graphs import article_scenario_graph
from backward_cgm_pd.graph import (
    ColoredGraph,
    Edge,
    full_edges,
    graph_key,
    out_edges,
    tau,
)
from backward_cgm_pd.io import load_simulated_datasets, write_json
from backward_cgm_pd.metrics import recovery_metrics
from backward_cgm_pd.search_submodel import backward_submodel
from backward_cgm_pd.search_tau import backward_cgm_pd

SCENARIO_CODE = {"A": "11", "B": "22"}
SEARCHES = {"tau": backward_cgm_pd, "submodel": backward_submodel}


def saved_data_path(scenario: str, p: str) -> Path:
    return (
        REPO_ROOT
        / "data"
        / "simulated-data"
        / f"simdf_{SCENARIO_CODE[scenario]}_{p}.RData"
    )


def symmetric_edges(graph: ColoredGraph) -> frozenset[Edge]:
    """Left representatives of twin edge pairs that share a single colour."""
    return out_edges(graph.E, graph.p).TE - graph.E_atomic


def symmetric_vertices(graph: ColoredGraph) -> frozenset[int]:
    """Left vertices whose diagonal element is tied to that of its twin."""
    return frozenset(range(1, graph.p // 2 + 1)) - graph.L_atomic


def _fit_one(
    data: np.ndarray,
    indices: np.ndarray,
    *,
    method: str,
    alpha: float,
    itmax: float,
    rcon_backend: str,
) -> dict[str, object] | None:
    """Fit one bootstrap resample; return ``None`` when the search fails."""
    resample = data[indices, :]
    search = SEARCHES[method]
    started = time.perf_counter()
    try:
        result = search(
            resample,
            alpha=alpha,
            itmax=int(itmax),
            rcon_backend=rcon_backend,
            n_jobs=1,
        )
    except Exception as error:  # numerically degenerate resample
        return {"failed": True, "error": f"{type(error).__name__}: {error}"}
    return {
        "failed": False,
        "runtime_seconds": time.perf_counter() - started,
        "iterations": result.iterations,
        "number_models": result.number_models,
        "pvalue": result.pvalue,
        "converged": bool(result.fit.converged),
        "saturated": result.iterations == 0,
        "model": result.model,
    }


def bootstrap_stability(
    data: np.ndarray,
    *,
    method: str,
    bootstrap: int,
    alpha: float,
    itmax: int,
    rcon_backend: str,
    seed: int,
    n_jobs: int,
    verbose: bool,
) -> dict[str, object]:
    n, p = data.shape
    search = SEARCHES[method]

    started = time.perf_counter()
    point = search(
        data, alpha=alpha, itmax=itmax, rcon_backend=rcon_backend, n_jobs=1
    )
    point_runtime = time.perf_counter() - started
    if verbose:
        print(
            f"point estimate: {point_runtime:.2f}s |E|={len(point.model.E)} "
            f"iterations={point.iterations}",
            flush=True,
        )

    rng = np.random.default_rng(seed)
    all_indices = [rng.integers(0, n, n) for _ in range(bootstrap)]

    outcomes = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(_fit_one)(
            data,
            indices,
            method=method,
            alpha=alpha,
            itmax=itmax,
            rcon_backend=rcon_backend,
        )
        for indices in all_indices
    )

    successes = [row for row in outcomes if row and not row["failed"]]
    failures = [row for row in outcomes if row and row["failed"]]
    if not successes:
        raise RuntimeError("every bootstrap resample failed")

    universe = sorted(full_edges(p).FV)
    left_vertices = list(range(1, p // 2 + 1))
    left_pairs = sorted(
        edge for edge in universe if edge[0] <= p // 2 and edge[1] <= p // 2
    ) + sorted(
        edge
        for edge in universe
        if edge[0] <= p // 2 < edge[1] and edge[1] != tau(edge[0], p)
    )

    edge_counter: Counter[Edge] = Counter()
    symmetry_counter: Counter[Edge] = Counter()
    vertex_counter: Counter[int] = Counter()
    model_counter: Counter[tuple[object, ...]] = Counter()
    for row in successes:
        model: ColoredGraph = row["model"]  # type: ignore[assignment]
        edge_counter.update(model.E)
        symmetry_counter.update(symmetric_edges(model))
        vertex_counter.update(symmetric_vertices(model))
        model_counter[graph_key(model)] += 1

    total = len(successes)
    edge_selection = {
        f"{i}-{j}": edge_counter[(i, j)] / total for i, j in universe
    }
    symmetry_selection = {
        f"{i}-{j}": symmetry_counter[(i, j)] / total for i, j in left_pairs
    }
    vertex_selection = {
        str(v): vertex_counter[v] / total for v in left_vertices
    }

    frequencies = np.array(list(edge_selection.values()))
    instability = float(np.mean(2 * frequencies * (1 - frequencies)))
    exact = model_counter[graph_key(point.model)] / total

    agreement = [
        recovery_metrics(row["model"], point.model).to_dict()  # type: ignore[arg-type]
        for row in successes
    ]
    agreement_mean = {
        key: float(np.nanmean([row[key] for row in agreement]))
        for key in agreement[0]
    }

    return {
        "point_estimate": {
            "model": point.model,
            "runtime_seconds": point_runtime,
            "iterations": point.iterations,
            "number_models": point.number_models,
            "pvalue": point.pvalue,
            "number_edges": len(point.model.E),
            "number_symmetries": len(symmetric_edges(point.model)),
            "number_parameters": point.model.n_parameters,
        },
        "bootstrap": {
            "requested": bootstrap,
            "successful": total,
            "failed": len(failures),
            "saturated": sum(1 for row in successes if row["saturated"]),
            "mean_runtime": float(
                np.mean([float(row["runtime_seconds"]) for row in successes])
            ),
        },
        "instability": instability,
        "exact_recovery_rate": exact,
        "mean_agreement_with_point_estimate": agreement_mean,
        "edge_selection": edge_selection,
        "symmetry_selection": symmetry_selection,
        "vertex_symmetry_selection": vertex_selection,
        "distinct_models": len(model_counter),
        "failures": [row["error"] for row in failures][:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["A", "B"], required=True)
    parser.add_argument("--p", choices=["8", "12", "16", "20"], required=True)
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="1-based index of the simulated dataset to analyse",
    )
    parser.add_argument("--method", choices=list(SEARCHES), default="tau")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--itmax", type=int, default=500)
    parser.add_argument(
        "--rcon-backend", choices=["mle", "grc_ipms"], default="mle"
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    datasets = load_simulated_datasets(saved_data_path(args.scenario, args.p))
    if not 1 <= args.replicate <= len(datasets):
        parser.error(
            f"--replicate must be within 1..{len(datasets)} for this configuration"
        )
    data = datasets[args.replicate - 1]
    truth = article_scenario_graph(args.scenario, int(args.p))

    result = bootstrap_stability(
        data,
        method=args.method,
        bootstrap=args.bootstrap,
        alpha=args.alpha,
        itmax=args.itmax,
        rcon_backend=args.rcon_backend,
        seed=args.seed,
        n_jobs=args.parallel,
        verbose=args.verbose,
    )
    point_model = result["point_estimate"]["model"]  # type: ignore[index]
    result["settings"] = {
        "scenario": args.scenario,
        "p": int(args.p),
        "replicate": args.replicate,
        "method": args.method,
        "bootstrap": args.bootstrap,
        "alpha": args.alpha,
        "itmax": args.itmax,
        "rcon_backend": args.rcon_backend,
        "seed": args.seed,
        "n_observations": int(data.shape[0]),
    }
    result["truth"] = truth
    result["point_estimate_vs_truth"] = recovery_metrics(  # type: ignore[index]
        point_model, truth
    ).to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(result, args.output)

    rows = [
        {"quantity": "edge", "label": key, "selection_frequency": value}
        for key, value in result["edge_selection"].items()  # type: ignore[union-attr]
    ] + [
        {"quantity": "edge_symmetry", "label": key, "selection_frequency": value}
        for key, value in result["symmetry_selection"].items()  # type: ignore[union-attr]
    ] + [
        {"quantity": "vertex_symmetry", "label": key, "selection_frequency": value}
        for key, value in result["vertex_symmetry_selection"].items()  # type: ignore[union-attr]
    ]
    frame = pd.DataFrame(rows)
    csv_path = args.output.with_name(f"{args.output.stem}-selection.csv")
    frame.to_csv(csv_path, index=False)

    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")
    print(
        f"instability={result['instability']:.4f}  "
        f"exact_recovery_rate={result['exact_recovery_rate']:.3f}  "
        f"distinct_models={result['distinct_models']}  "
        f"failed={result['bootstrap']['failed']}"  # type: ignore[index]
    )


if __name__ == "__main__":
    main()
