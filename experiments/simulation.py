"""Reproduce the synthetic-data comparison from simulation/simulation.R.

Examples
--------
Quick smoke run on one saved dataset::

    python experiments/simulation.py --scenario A --p 8 --replicates 1 --method tau

Full experiment (computationally expensive)::

    python experiments/simulation.py --scenario all --p all --replicates 20 --method both
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PORT_ROOT = Path(__file__).resolve().parents[1]
# Trong kho mã nguồn gốc, mã Python nằm trong python-port/ nên dữ liệu ở thư mục
# cha; ở kho này mã nằm ngay gốc kho, nên hai đường dẫn trùng nhau.
REPO_ROOT = PORT_ROOT.parent if (PORT_ROOT.parent / "applications").is_dir() else PORT_ROOT
sys.path.insert(0, str(PORT_ROOT / "src"))

from backward_cgm_pd.article_graphs import article_scenario_graph
from backward_cgm_pd.graph import ColoredGraph, full_edges, out_edges, tau_edges
from backward_cgm_pd.io import load_simulated_datasets, read_json, write_json
from backward_cgm_pd.metrics import RecoveryMetrics, average_metrics, recovery_metrics
from backward_cgm_pd.rcon import fit_rcon_grc_ipms
from backward_cgm_pd.search_submodel import backward_submodel
from backward_cgm_pd.search_tau import backward_cgm_pd


# pair_count, atomic_pair_count, transverse_count, extra_single_count, L_atomic_count
SCENARIOS = {
    "A": {
        8: (2, 1, 0, 1, 3),
        12: (5, 4, 1, 2, 4),
        16: (9, 7, 1, 3, 6),
        20: (14, 11, 2, 4, 8),
    },
    "B": {
        8: (4, 1, 0, 2, 1),
        12: (9, 3, 1, 4, 2),
        16: (15, 5, 3, 9, 2),
        20: (24, 8, 6, 12, 2),
    },
}
CHECKPOINT_VERSION = 3


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def make_scenario_graph(scenario: str, p: int, seed: int = 2024) -> ColoredGraph:
    """Create a statistically equivalent scenario with deterministic NumPy RNG.

    The original saved R samples should be used for exact article inputs.  This
    generator uses the same counts/densities but cannot be bit-identical to R's
    RNG stream.
    """
    pair_count, atomic_count, transverse_count, extra_count, left_count = SCENARIOS[
        scenario
    ][p]
    rng = np.random.default_rng(seed + p + (0 if scenario == "A" else 10_000))
    universe = full_edges(p)
    pair_representatives = sorted(universe.FL)
    chosen_pairs = {
        pair_representatives[index]
        for index in rng.choice(len(pair_representatives), pair_count, replace=False)
    }
    atomic = {
        sorted(chosen_pairs)[index]
        for index in rng.choice(pair_count, atomic_count, replace=False)
    }
    chosen_transverse = {
        sorted(universe.FT)[index]
        for index in rng.choice(p // 2, transverse_count, replace=False)
    }
    paired_full = chosen_pairs | tau_edges(chosen_pairs, p)
    rest = sorted(universe.FV - paired_full - universe.FT)
    extra = {
        rest[index] for index in rng.choice(len(rest), extra_count, replace=False)
    }
    left_atomic = {
        int(index) + 1
        for index in rng.choice(p // 2, left_count, replace=False)
    }
    return ColoredGraph(
        p,
        frozenset(left_atomic),
        frozenset(paired_full | chosen_transverse | extra),
        frozenset(atomic),
    )


def graph_from_json(value: dict[str, object]) -> ColoredGraph:
    return ColoredGraph.from_r(
        int(value["p"]),
        value.get("L.as"),
        value.get("E"),
        value.get("E.as"),
    )


def update_method_summary(method_output: dict[str, object]) -> None:
    rows: list[dict[str, object]] = method_output["runs"]
    metric_objects = [
        RecoveryMetrics(**row["metrics"]) for row in rows if "metrics" in row
    ]
    edge_counts = [int(row["number_edges"]) for row in rows]
    symmetry_counts = [int(row["number_symmetries"]) for row in rows]
    mean_metrics = average_metrics(metric_objects)
    method_output.update(
        {
            "runs": rows,
            "mean_runtime_seconds": float(
                np.mean([row["runtime_seconds"] for row in rows])
            ),
            "mean_iterations": float(
                np.mean([row["iterations"] for row in rows])
            ),
            "mean_number_models": float(
                np.mean([row["number_models"] for row in rows])
            ),
            "mean_number_edges": float(np.mean(edge_counts)),
            "sd_number_edges": (
                float(np.std(edge_counts, ddof=1)) if len(edge_counts) > 1 else 0.0
            ),
            "mean_number_symmetries": float(np.mean(symmetry_counts)),
            "sd_number_symmetries": (
                float(np.std(symmetry_counts, ddof=1))
                if len(symmetry_counts) > 1
                else 0.0
            ),
            "mean_metrics": mean_metrics,
            "mean_metrics_percent": {
                key: 100.0 * value for key, value in mean_metrics.items()
            },
        }
    )


def update_run_derived(
    row: dict[str, object], truth: ColoredGraph | None
) -> None:
    model = graph_from_json(row["model"])
    row["number_edges"] = len(model.E)
    row["number_symmetries"] = len(
        out_edges(model.E, model.p).TE - model.E_atomic
    )
    if truth is not None:
        row["metrics"] = recovery_metrics(model, truth).to_dict()


def generate_datasets(
    scenario: str,
    p: int,
    *,
    replicates: int = 20,
    n: int = 100,
    rho: float = 0.5,
    seed: int = 2024,
) -> tuple[ColoredGraph, list[np.ndarray]]:
    graph = make_scenario_graph(scenario, p, seed)
    equicovariance = np.full((p, p), rho)
    np.fill_diagonal(equicovariance, 1.0)
    target = fit_rcon_grc_ipms(
        None, graph, covariance=equicovariance, n_observations=n
    )
    covariance = np.linalg.inv(target.precision)
    rng = np.random.default_rng(seed + 100 * p + (0 if scenario == "A" else 1))
    datasets = [
        rng.multivariate_normal(np.zeros(p), covariance, size=n)
        for _ in range(replicates)
    ]
    return graph, datasets


def saved_data_path(scenario: str, p: int) -> Path:
    prefix = "11" if scenario == "A" else "22"
    return REPO_ROOT / "simulation" / "simulated-data" / f"simdf_{prefix}_{p}.RData"


def run_configuration(
    scenario: str,
    p: int,
    *,
    source: str,
    replicates: int,
    method: str,
    alpha: float,
    itmax: int,
    parallel: int,
    rcon_backend: str,
    verbose: bool,
    existing: dict[str, object] | None = None,
    checkpoint: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    truth: ColoredGraph | None
    if source == "saved":
        datasets = load_simulated_datasets(saved_data_path(scenario, p))[:replicates]
        truth = article_scenario_graph(scenario, p)
    else:
        truth, datasets = generate_datasets(
            scenario, p, replicates=replicates
        )

    methods = (
        ("tau", backward_cgm_pd),
        ("submodel", backward_submodel),
    )
    if method != "both":
        methods = tuple(item for item in methods if item[0] == method)
    method_names = tuple(name for name, _ in methods)

    output: dict[str, object]
    if existing is None:
        output = {
            "scenario": scenario,
            "p": p,
            "source": source,
            "replicates": len(datasets),
            "note": (
                "Saved article datasets with the true graph reconstructed from simulation.R."
                if source == "saved"
                else "New statistically equivalent datasets generated with NumPy RNG."
            ),
            "true_model": truth.to_dict() if truth is not None else None,
            "methods": {},
        }
    else:
        output = existing
    output["true_model"] = truth.to_dict() if truth is not None else None
    if source == "saved":
        output["note"] = (
            "Saved article datasets with the true graph reconstructed from "
            "simulation.R (interpreting its FI typo as fullEdges()$FT)."
        )

    total_units = len(methods) * len(datasets)

    def completed_units() -> int:
        return sum(
            len(output["methods"].get(method_name, {}).get("runs", []))
            for method_name in method_names
        )

    print(
        f"[{_timestamp()}] [Simulation] configuration scenario={scenario} p={p} "
        f"| methods={','.join(method_names)} | replicates={len(datasets)} "
        f"| completed={completed_units()}/{total_units}",
        flush=True,
    )

    for method_position, (name, function) in enumerate(methods, 1):
        method_output = output["methods"].setdefault(name, {"runs": []})
        rows: list[dict[str, object]] = method_output["runs"]
        # Upgrade older checkpoints without rerunning completed replicates.
        if truth is not None:
            for row in rows:
                if "model" in row:
                    update_run_derived(row, truth)
            if rows:
                update_method_summary(method_output)
                if checkpoint is not None:
                    checkpoint(output)
        completed = {int(row["replicate"]) for row in rows}
        for index, data in enumerate(datasets, 1):
            if index in completed:
                print(
                    f"[{_timestamp()}] [Simulation] RESUME skip "
                    f"method={name} replicate={index}/{len(datasets)}",
                    flush=True,
                )
                continue
            done_before = completed_units()
            percent_before = 100 * done_before / total_units
            print(
                f"[{_timestamp()}] [Simulation] START method={name} "
                f"({method_position}/{len(methods)}) "
                f"replicate={index}/{len(datasets)} | "
                f"overall={done_before}/{total_units} ({percent_before:.1f}%)",
                flush=True,
            )
            started = perf_counter()
            result = function(
                data,
                itmax=itmax,
                alpha=alpha,
                n_jobs=parallel,
                rcon_backend=rcon_backend,
                verbose=verbose,
            )
            row: dict[str, object] = {
                "replicate": index,
                "runtime_seconds": perf_counter() - started,
                "iterations": result.iterations,
                "number_models": result.number_models,
                "pvalue": result.pvalue,
                "model": result.model.to_dict(),
            }
            update_run_derived(row, truth)
            rows.append(row)
            done_after = completed_units()
            runtimes = [
                float(run["runtime_seconds"])
                for method_name in method_names
                for run in output["methods"].get(method_name, {}).get("runs", [])
            ]
            mean_runtime = float(np.mean(runtimes))
            eta = mean_runtime * (total_units - done_after)
            print(
                f"[{_timestamp()}] [Simulation] DONE  method={name} "
                f"replicate={index}/{len(datasets)} "
                f"| elapsed={_duration(float(row['runtime_seconds']))} "
                f"| overall={done_after}/{total_units} "
                f"({100 * done_after / total_units:.1f}%) "
                f"| ETA~{_duration(eta)}",
                flush=True,
            )
            update_method_summary(method_output)
            if checkpoint is not None:
                checkpoint(output)

        update_method_summary(method_output)
    print(
        f"[{_timestamp()}] [Simulation] COMPLETE scenario={scenario} p={p} "
        f"| {completed_units()}/{total_units} units",
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["A", "B", "all"], default="all")
    parser.add_argument("--p", choices=["8", "12", "16", "20", "all"], default="all")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--method", choices=["tau", "submodel", "both"], default="both")
    parser.add_argument("--source", choices=["saved", "generate"], default="saved")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--itmax", type=int, default=500)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument(
        "--rcon-backend",
        choices=["grc_ipms", "mle"],
        default="grc_ipms",
        help="RCON fitter. grc_ipms ports gRc::rcox(method='ipms').",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed replicates from an existing output JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PORT_ROOT / "python-results" / "simulation.json",
    )
    arguments = parser.parse_args()

    scenarios = ["A", "B"] if arguments.scenario == "all" else [arguments.scenario]
    dimensions = [8, 12, 16, 20] if arguments.p == "all" else [int(arguments.p)]
    settings = {
        "scenarios": scenarios,
        "dimensions": dimensions,
        "replicates": arguments.replicates,
        "method": arguments.method,
        "source": arguments.source,
        "alpha": arguments.alpha,
        "itmax": arguments.itmax,
        "parallel": arguments.parallel,
        "rcon_backend": arguments.rcon_backend,
    }
    state: dict[str, object] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "settings": settings,
        "complete": False,
        "configurations": [],
    }
    if arguments.resume and arguments.output.exists():
        candidate = read_json(arguments.output)
        if (
            candidate.get("checkpoint_version") == CHECKPOINT_VERSION
            and candidate.get("settings") == settings
        ):
            state = candidate
            print(f"Resuming checkpoint {arguments.output}")
        else:
            print("Ignoring incompatible checkpoint and starting a fresh run.")

    results: list[dict[str, object]] = state["configurations"]

    def save_checkpoint(_: dict[str, object] | None = None) -> None:
        state["complete"] = False
        write_json(state, arguments.output)
        print(
            f"[{_timestamp()}] [Simulation] CHECKPOINT {arguments.output}",
            flush=True,
        )

    for scenario in scenarios:
        for p in dimensions:
            existing = next(
                (
                    item
                    for item in results
                    if item["scenario"] == scenario and int(item["p"]) == p
                ),
                None,
            )
            if existing is None:
                existing = {
                    "scenario": scenario,
                    "p": p,
                    "source": arguments.source,
                    "replicates": arguments.replicates,
                    "note": (
                        "Saved article datasets with the true graph reconstructed from simulation.R."
                        if arguments.source == "saved"
                        else "New statistically equivalent datasets generated with NumPy RNG."
                    ),
                    "methods": {},
                }
                results.append(existing)
                save_checkpoint()
            run_configuration(
                scenario,
                p,
                source=arguments.source,
                replicates=arguments.replicates,
                method=arguments.method,
                alpha=arguments.alpha,
                itmax=arguments.itmax,
                parallel=arguments.parallel,
                rcon_backend=arguments.rcon_backend,
                verbose=arguments.verbose,
                existing=existing,
                checkpoint=save_checkpoint,
            )
    state["complete"] = True
    write_json(state, arguments.output)
    print(
        f"[{_timestamp()}] [Simulation] ALL REQUESTED CONFIGURATIONS COMPLETE",
        flush=True,
    )
    summary_rows: list[dict[str, object]] = []
    for configuration in results:
        for method_name, method_result in configuration["methods"].items():
            row: dict[str, object] = {
                "scenario": configuration["scenario"],
                "p": configuration["p"],
                "method": method_name,
                "mean_runtime_seconds": method_result["mean_runtime_seconds"],
                "mean_iterations": method_result["mean_iterations"],
                "mean_number_models": method_result["mean_number_models"],
                "mean_number_edges": method_result["mean_number_edges"],
                "sd_number_edges": method_result["sd_number_edges"],
                "mean_number_symmetries": method_result[
                    "mean_number_symmetries"
                ],
                "sd_number_symmetries": method_result[
                    "sd_number_symmetries"
                ],
            }
            row.update(method_result["mean_metrics"])
            row.update(
                {
                    f"{key}_percent": value
                    for key, value in method_result[
                        "mean_metrics_percent"
                    ].items()
                }
            )
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = arguments.output.with_name(arguments.output.stem + "-summary.csv")
    summary.to_csv(summary_path, index=False)
    if len(summary):
        for scenario in sorted(summary["scenario"].unique()):
            subset = summary[summary["scenario"] == scenario]
            figure, axes = plt.subplots(1, 2, figsize=(11, 4))
            for method_name in sorted(subset["method"].unique()):
                rows = subset[subset["method"] == method_name].sort_values("p")
                axes[0].plot(
                    rows["p"], rows["mean_runtime_seconds"], "o-", label=method_name
                )
                axes[1].plot(
                    rows["p"], rows["mean_number_models"], "o-", label=method_name
                )
            axes[0].set(xlabel="p", ylabel="average seconds", title=f"Scenario {scenario}")
            axes[1].set(
                xlabel="p",
                ylabel="average fitted models",
                title=f"Scenario {scenario}",
            )
            for axis in axes:
                axis.legend()
                axis.grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(
                arguments.output.with_name(
                    arguments.output.stem + f"-scenario-{scenario}.png"
                ),
                dpi=180,
            )
            plt.close(figure)
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
