"""Reconstruct Table 3.2 from the authors' archived R simulation outputs.

This performs no model fitting.  It is a fast, transparent recomputation of
the table statistics from ``simulation/simulation-results``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parent
sys.path.insert(0, str(PORT_ROOT / "src"))

from backward_cgm_pd.article_graphs import article_scenario_graph
from backward_cgm_pd.graph import out_edges
from backward_cgm_pd.io import graph_from_r_mapping, read_rdata
from backward_cgm_pd.metrics import average_metrics, recovery_metrics


def _single_r_object(path: Path):
    objects = read_rdata(path)
    if len(objects) != 1:
        raise ValueError(f"Expected one R object in {path}, found {list(objects)}")
    return next(iter(objects.values()))


def _sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def archived_configuration(
    results_root: Path, scenario: str, p: int, method: str
) -> dict[str, object]:
    prefix = "11" if scenario == "A" else "22"
    directory = results_root / f"{prefix}-{p}"
    outputs = _single_r_object(directory / f"out_{prefix}_{p}_{method}.RData")
    graphs = [graph_from_r_mapping(value, p=p) for value in outputs]
    truth = article_scenario_graph(scenario, p)
    metrics = average_metrics(
        [recovery_metrics(graph, truth) for graph in graphs]
    )

    edge_counts = [float(len(graph.E)) for graph in graphs]
    symmetry_counts = [
        float(len(out_edges(graph.E, p).TE - graph.E_atomic))
        for graph in graphs
    ]
    runtimes = np.asarray(
        _single_r_object(directory / f"runtime_{prefix}_{p}_{method}.RData"),
        dtype=float,
    ).reshape(-1)
    model_counts = np.asarray(
        _single_r_object(directory / f"nmodels_{prefix}_{p}_{method}.RData"),
        dtype=float,
    ).reshape(-1)
    iterations = np.asarray(
        _single_r_object(directory / f"nsteps_{prefix}_{p}_{method}.RData"),
        dtype=float,
    ).reshape(-1)

    row: dict[str, object] = {
        "scenario": scenario,
        "p": p,
        "order": "twin" if method == "tau" else "submodel",
        "method": method,
        "replicates": len(graphs),
        "mean_number_edges": float(np.mean(edge_counts)),
        "sd_number_edges": _sample_sd(edge_counts),
        "mean_number_symmetries": float(np.mean(symmetry_counts)),
        "sd_number_symmetries": _sample_sd(symmetry_counts),
        "mean_runtime_seconds": float(np.mean(runtimes)),
        "mean_iterations": float(np.mean(iterations)),
        "mean_number_models": float(np.mean(model_counts)),
    }
    row.update({f"{key}_percent": 100.0 * value for key, value in metrics.items()})
    return row


def formatted_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[
        [
            "scenario",
            "p",
            "order",
            "mean_number_edges",
            "sd_number_edges",
            "ePPV_percent",
            "eTPR_percent",
            "eTNR_percent",
            "mean_number_symmetries",
            "sd_number_symmetries",
            "sPPV_percent",
            "sTPR_percent",
            "sTNR_percent",
            "mean_runtime_seconds",
            "mean_number_models",
        ]
    ].copy()
    result["#edges"] = result.apply(
        lambda row: f"{row.mean_number_edges:.0f}({row.sd_number_edges:.0f})",
        axis=1,
    )
    result["#sym"] = result.apply(
        lambda row: (
            f"{row.mean_number_symmetries:.0f}"
            f"({row.sd_number_symmetries:.0f})"
        ),
        axis=1,
    )
    result["Time(s)"] = result["mean_runtime_seconds"].round().astype(int)
    result["#models"] = result["mean_number_models"].round().astype(int)
    result = result.rename(
        columns={
            "ePPV_percent": "ePPV%",
            "eTPR_percent": "eTPR%",
            "eTNR_percent": "eTNR%",
            "sPPV_percent": "sPPV%",
            "sTPR_percent": "sTPR%",
            "sTNR_percent": "sTNR%",
        }
    )
    columns = [
        "scenario",
        "p",
        "order",
        "#edges",
        "ePPV%",
        "eTPR%",
        "eTNR%",
        "#sym",
        "sPPV%",
        "sTPR%",
        "sTNR%",
        "Time(s)",
        "#models",
    ]
    result = result[columns]
    percent_columns = [
        "ePPV%",
        "eTPR%",
        "eTNR%",
        "sPPV%",
        "sTPR%",
        "sTNR%",
    ]
    result[percent_columns] = result[percent_columns].round(2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "simulation" / "simulation-results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PORT_ROOT / "python-results" / "paper-table",
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        archived_configuration(arguments.results_root, scenario, p, method)
        for scenario in ("A", "B")
        for p in (8, 12, 16, 20)
        for method in ("tau", "submod")
    ]
    raw = pd.DataFrame(rows)
    table = formatted_table(raw)
    raw.to_csv(arguments.output_dir / "table-3.2-raw.csv", index=False)
    table.to_csv(arguments.output_dir / "table-3.2-paper-format.csv", index=False)
    (arguments.output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source": "authors_archived_R_outputs",
                "model_refit_performed": False,
                "results_root": str(arguments.results_root),
                "description": (
                    "Metrics recomputed from the selected models, runtimes, "
                    "iterations and fitted-model counts committed by the authors."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print(f"Wrote {arguments.output_dir}")


if __name__ == "__main__":
    main()
