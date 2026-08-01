"""Head-to-head comparison of the two RCON fitting backends.

The reproduction runs the small configurations (``p = 8, 12``) with the
``grc_ipms`` backend, which mirrors how the original R code fits candidate
models, and the large ones (``p = 16, 20``) with the faster ``mle`` backend.
Because no configuration was ever run both ways on the same machine, the
runtimes reported for the two groups are not directly comparable, and the
report says so explicitly.

This script closes that gap. For each configuration it runs the identical
search twice - same data, same replicates, same worker count, same machine -
changing only the backend. That isolates the effect of the fitting procedure
from every other factor, which is what an ablation is meant to do.

Two questions are answered:

1. How much faster is ``mle`` than ``grc_ipms``?
2. Do the two backends select the same models? If they diverge, the runtime
   comparison in the report would rest on procedures that are not doing the
   same thing.

Usage::

    python experiments/ablation_backend.py --scenario A --p 8 \
        --replicates 10 --output results/ablation-A-p8.json
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
from backward_cgm_pd.graph import graph_key
from backward_cgm_pd.io import load_simulated_datasets, write_json
from backward_cgm_pd.metrics import average_metrics, recovery_metrics
from backward_cgm_pd.search_submodel import backward_submodel
from backward_cgm_pd.search_tau import backward_cgm_pd

SCENARIO_CODE = {"A": "11", "B": "22"}
SEARCHES = {"tau": backward_cgm_pd, "submodel": backward_submodel}
BACKENDS = ("grc_ipms", "mle")


def saved_data_path(scenario: str, p: str) -> Path:
    return (
        REPO_ROOT
        / "simulation"
        / "simulated-data"
        / f"simdf_{SCENARIO_CODE[scenario]}_{p}.RData"
    )


def run_ablation(
    scenario: str,
    p: str,
    *,
    method: str,
    replicates: int,
    alpha: float,
    itmax: int,
    parallel: int,
    verbose: bool,
) -> dict[str, object]:
    datasets = load_simulated_datasets(saved_data_path(scenario, p))[:replicates]
    truth = article_scenario_graph(scenario, int(p))
    search = SEARCHES[method]

    rows: list[dict[str, object]] = []
    for index, data in enumerate(datasets, 1):
        selected: dict[str, object] = {}
        for backend in BACKENDS:
            started = time.perf_counter()
            result = search(
                data,
                alpha=alpha,
                itmax=itmax,
                rcon_backend=backend,
                n_jobs=parallel,
            )
            runtime = time.perf_counter() - started
            selected[backend] = result.model
            rows.append(
                {
                    "replicate": index,
                    "backend": backend,
                    "runtime_seconds": runtime,
                    "iterations": result.iterations,
                    "number_models": result.number_models,
                    "pvalue": result.pvalue,
                    "number_edges": len(result.model.E),
                    "model": result.model,
                    "metrics": recovery_metrics(result.model, truth).to_dict(),
                }
            )
            if verbose:
                print(
                    f"[{scenario}/p={p}] replicate {index}/{len(datasets)} "
                    f"{backend}: {runtime:.2f}s |E|={len(result.model.E)}",
                    flush=True,
                )
        same = graph_key(selected["grc_ipms"]) == graph_key(selected["mle"])
        rows[-1]["same_model_as_other_backend"] = same
        rows[-2]["same_model_as_other_backend"] = same

    summary: list[dict[str, object]] = []
    for backend in BACKENDS:
        subset = [row for row in rows if row["backend"] == backend]
        runtimes = [float(row["runtime_seconds"]) for row in subset]
        summary.append(
            {
                "scenario": scenario,
                "p": int(p),
                "method": method,
                "backend": backend,
                "replicates": len(subset),
                "mean_runtime": float(np.mean(runtimes)),
                "sd_runtime": float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0,
                "mean_iterations": float(np.mean([r["iterations"] for r in subset])),
                "mean_number_models": float(np.mean([r["number_models"] for r in subset])),
                "mean_number_edges": float(np.mean([r["number_edges"] for r in subset])),
                **average_metrics(
                    [recovery_metrics(r["model"], truth) for r in subset]
                ),
            }
        )

    ipms_time = summary[0]["mean_runtime"]
    mle_time = summary[1]["mean_runtime"]
    agreement = [row for row in rows if row["backend"] == "mle"]
    n_same = sum(1 for row in agreement if row.get("same_model_as_other_backend"))

    return {
        "settings": {
            "scenario": scenario,
            "p": int(p),
            "method": method,
            "replicates": len(datasets),
            "alpha": alpha,
            "itmax": itmax,
            "parallel": parallel,
        },
        "truth": truth,
        "speedup_mle_over_ipms": float(ipms_time / mle_time) if mle_time else None,
        "identical_models": n_same,
        "identical_model_rate": n_same / len(agreement) if agreement else None,
        "runs": rows,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["A", "B"], required=True)
    parser.add_argument("--p", choices=["8", "12", "16", "20"], required=True)
    parser.add_argument("--method", choices=list(SEARCHES), default="tau")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--itmax", type=int, default=500)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run_ablation(
        args.scenario,
        args.p,
        method=args.method,
        replicates=args.replicates,
        alpha=args.alpha,
        itmax=args.itmax,
        parallel=args.parallel,
        verbose=args.verbose,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(result, args.output)
    frame = pd.DataFrame(result["summary"])
    frame.to_csv(args.output.with_name(f"{args.output.stem}-summary.csv"), index=False)

    print(f"Wrote {args.output}")
    print(frame.to_string(index=False))
    print(
        f"\nspeedup (ipms/mle) = {result['speedup_mle_over_ipms']:.2f}x  |  "
        f"mô hình trùng nhau: {result['identical_models']}/"
        f"{result['settings']['replicates']}"
    )


if __name__ == "__main__":
    main()
