"""Baseline comparison on the simulated data of Roverato & Nguyen (2024).

The article evaluates two search strategies on the twin lattice (``tau``) and
on the model inclusion lattice (``submodel``).  Both belong to the proposed
method, so the simulation study of the paper contains no external competitor.

This script adds two baselines that are standard in the colored graphical
model literature and that the article itself uses for the air quality data:

``pdglasso-cov``
    pdRCON graphical lasso applied to the sample covariance matrix.

``pdglasso-cor``
    the same estimator applied to the sample correlation matrix, i.e. after
    standardising every variable.  The article stresses that pdglasso is not
    scale invariant, so the two variants are genuinely different baselines.

Both are scored with the recovery metrics of Table 3.2, exactly the ones used
for the stepwise procedures, which makes the comparison directly readable
against the numbers reproduced in Chapter 4 of the report.

Usage::

    python experiments/baseline_simulation.py --scenario A --p 8 \
        --replicates 20 --output results/baseline-A-p8.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from backward_cgm_pd.article_graphs import article_scenario_graph
from backward_cgm_pd.io import load_simulated_datasets, write_json
from backward_cgm_pd.metrics import average_metrics, recovery_metrics
from backward_cgm_pd.pdglasso import select_pdglasso

SCENARIO_CODE = {"A": "11", "B": "22"}


def saved_data_path(scenario: str, p: str) -> Path:
    return (
        REPO_ROOT
        / "data"
        / "simulated-data"
        / f"simdf_{SCENARIO_CODE[scenario]}_{p}.RData"
    )


def correlation_from_data(data: np.ndarray) -> np.ndarray:
    """Sample correlation matrix, i.e. the covariance of standardised data."""
    return np.corrcoef(data, rowvar=False)


def run_baseline(
    scenario: str,
    p: str,
    *,
    replicates: int,
    points: int,
    gamma_ebic: float,
    verbose: bool,
) -> dict[str, object]:
    datasets = load_simulated_datasets(saved_data_path(scenario, p))[:replicates]
    truth = article_scenario_graph(scenario, int(p))

    rows: list[dict[str, object]] = []
    for index, data in enumerate(datasets, 1):
        n = data.shape[0]
        for variant, matrix in (
            ("pdglasso-cov", np.cov(data, rowvar=False, ddof=1)),
            ("pdglasso-cor", correlation_from_data(data)),
        ):
            started = time.perf_counter()
            selection = select_pdglasso(
                matrix, n, points=points, gamma_ebic=gamma_ebic
            )
            runtime = time.perf_counter() - started
            metrics = recovery_metrics(selection.graph, truth)
            rows.append(
                {
                    "replicate": index,
                    "method": variant,
                    "runtime_seconds": runtime,
                    "lambda1": selection.best_lambdas[0],
                    "lambda2": selection.best_lambdas[1],
                    "number_edges": len(selection.graph.E),
                    "number_parameters": selection.graph.n_parameters,
                    "model": selection.graph,
                    "metrics": metrics.to_dict(),
                }
            )
            if verbose:
                print(
                    f"[{scenario}/p={p}] replicate {index}/{len(datasets)} "
                    f"{variant}: {runtime:.2f}s |E|={len(selection.graph.E)}",
                    flush=True,
                )

    summary: list[dict[str, object]] = []
    for variant in ("pdglasso-cov", "pdglasso-cor"):
        subset = [row for row in rows if row["method"] == variant]
        runtimes = [float(row["runtime_seconds"]) for row in subset]
        edges = [int(row["number_edges"]) for row in subset]
        averaged = average_metrics(
            [recovery_metrics(row["model"], truth) for row in subset]
        )
        summary.append(
            {
                "scenario": scenario,
                "p": int(p),
                "method": variant,
                "replicates": len(subset),
                "mean_runtime": float(np.mean(runtimes)),
                "sd_runtime": float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0,
                "mean_number_edges": float(np.mean(edges)),
                **averaged,
            }
        )

    return {
        "settings": {
            "scenario": scenario,
            "p": int(p),
            "replicates": len(datasets),
            "points": points,
            "gamma_ebic": gamma_ebic,
            "source": "saved",
        },
        "truth": truth,
        "runs": rows,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["A", "B"], required=True)
    parser.add_argument("--p", choices=["8", "12", "16", "20"], required=True)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--gamma-ebic", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run_baseline(
        args.scenario,
        args.p,
        replicates=args.replicates,
        points=args.points,
        gamma_ebic=args.gamma_ebic,
        verbose=args.verbose,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(result, args.output)
    summary_path = args.output.with_name(f"{args.output.stem}-summary.csv")
    pd.DataFrame(result["summary"]).to_csv(summary_path, index=False)
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")
    print(pd.DataFrame(result["summary"]).to_string(index=False))


if __name__ == "__main__":
    main()
