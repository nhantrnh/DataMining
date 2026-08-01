"""Recreate the fMRI graph figures and optionally refit supplied subject data."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

PORT_ROOT = Path(__file__).resolve().parents[1]
# Trong kho mã nguồn gốc, mã Python nằm trong python-port/ nên dữ liệu ở thư mục
# cha; ở kho này mã nằm ngay gốc kho, nên hai đường dẫn trùng nhau.
REPO_ROOT = PORT_ROOT.parent if (PORT_ROOT.parent / "applications").is_dir() else PORT_ROOT
sys.path.insert(0, str(PORT_ROOT / "src"))

from backward_cgm_pd.graph import ColoredGraph, full_edges, out_edges, tau_edges
from backward_cgm_pd.io import load_saved_search, read_json, write_json
from backward_cgm_pd.plotting import plot_colored_graph
from backward_cgm_pd.search_tau import backward_cgm_pd

CHECKPOINT_VERSION = 2


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def components(graph: ColoredGraph) -> dict[str, ColoredGraph]:
    p = graph.p
    q = p // 2
    parts = out_edges(graph.E, p)
    left_asymmetric = parts.EL - parts.TE
    right_asymmetric = parts.ER - tau_edges(parts.TE, p)
    symmetric = parts.TE - graph.E_atomic
    symmetric_full = symmetric | tau_edges(symmetric, p)
    within = frozenset(
        edge
        for edge in symmetric_full
        if (edge[0] <= q and edge[1] <= q)
        or (edge[0] > q and edge[1] > q)
    )
    between = symmetric_full - within - parts.ET
    return {
        "asymmetric-left": ColoredGraph(p, graph.L_atomic, left_asymmetric, frozenset()),
        "asymmetric-right": ColoredGraph(p, graph.L_atomic, right_asymmetric, frozenset()),
        "symmetric-within": ColoredGraph(p, graph.L_atomic, within, frozenset()),
        "symmetric-between": ColoredGraph(p, graph.L_atomic, between, frozenset()),
    }


def graph_summary(graph: ColoredGraph) -> dict[str, int]:
    parts = out_edges(graph.E, graph.p)
    symmetric = parts.TE - graph.E_atomic
    return {
        "number_edges": len(graph.E),
        "number_atomic_vertex_pairs": len(graph.L_atomic),
        "number_atomic_twin_edge_pairs": len(graph.E_atomic),
        "number_symmetric_twin_edge_pairs": len(symmetric),
        "number_transverse_edges": len(parts.ET),
        "number_asymmetric_left_edges": len(parts.EL - parts.TE),
        "number_asymmetric_right_edges": len(parts.ER - tau_edges(parts.TE, graph.p)),
        "number_parameters": graph.n_parameters,
    }


def load_data(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return np.loadtxt(path, delimiter=",")
    raise ValueError("Subject data must be .npy or .csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saved-dir",
        type=Path,
        default=REPO_ROOT
        / "applications"
        / "fMRIdata"
        / "output-fMRI"
        / "36variables",
    )
    parser.add_argument("--subject14-data", type=Path)
    parser.add_argument("--subject15-data", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--itmax", type=int, default=1_000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed subjects from an existing models JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PORT_ROOT / "python-results" / "fmri",
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "saved_dir": str(arguments.saved_dir),
        "subject14_data": (
            str(arguments.subject14_data) if arguments.subject14_data else None
        ),
        "subject15_data": (
            str(arguments.subject15_data) if arguments.subject15_data else None
        ),
        "alpha": arguments.alpha,
        "itmax": arguments.itmax,
    }
    checkpoint_path = arguments.output_dir / "models.json"
    summary: dict[str, object] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "settings": settings,
        "complete": False,
    }
    if arguments.resume and checkpoint_path.exists():
        candidate = read_json(checkpoint_path)
        if (
            candidate.get("checkpoint_version") == CHECKPOINT_VERSION
            and candidate.get("settings") == settings
        ):
            summary = candidate
            print(f"Resuming checkpoint {checkpoint_path}")
        else:
            print("Ignoring incompatible checkpoint and starting a fresh run.")

    def checkpoint() -> None:
        summary["complete"] = False
        write_json(summary, checkpoint_path)
        completed = sum(f"subject{subject}" in summary for subject in (14, 15))
        print(
            f"[{_timestamp()}] [fMRI] CHECKPOINT "
            f"{completed}/2 subjects ({50 * completed:.1f}%) "
            f"| {checkpoint_path}",
            flush=True,
        )

    checkpoint()
    supplied = {14: arguments.subject14_data, 15: arguments.subject15_data}
    for position, subject in enumerate((14, 15), 1):
        subject_key = f"subject{subject}"
        subject_started = perf_counter()
        source_name = "time-series refit" if supplied[subject] else "saved model"
        print(
            f"[{_timestamp()}] [fMRI] START Subject {subject} "
            f"({position}/2) | source={source_name}",
            flush=True,
        )
        if subject_key in summary:
            print(
                f"[{_timestamp()}] [fMRI] RESUME skip completed model "
                f"selection for Subject {subject}",
                flush=True,
            )
            model_mapping = summary[subject_key]["model"]
            graph = ColoredGraph.from_r(
                int(model_mapping["p"]),
                model_mapping["L.as"],
                model_mapping["E"],
                model_mapping["E.as"],
            )
        else:
            if supplied[subject]:
                data = load_data(supplied[subject])
                if data.shape[1] != 36:
                    raise ValueError(f"Subject {subject} data must have 36 columns")
                result = backward_cgm_pd(
                    data,
                    itmax=arguments.itmax,
                    alpha=arguments.alpha,
                    verbose=arguments.verbose,
                )
                graph = result.model
                summary[subject_key] = result.to_dict()
            else:
                path = arguments.saved_dir / f"res{subject}.36.tauc.RData"
                graph, metadata = load_saved_search(path, p=36)
                summary[subject_key] = {
                    "source": str(path),
                    "model": graph.to_dict(),
                    "metadata": metadata,
                }
            checkpoint()
        subject_components = components(graph)
        summary[subject_key]["graph_summary"] = graph_summary(graph)
        summary[subject_key]["component_edge_counts"] = {
            name: len(component.E)
            for name, component in subject_components.items()
        }
        checkpoint()
        print(
            f"[{_timestamp()}] [fMRI] RENDER Subject {subject} "
            f"| 4 graph components",
            flush=True,
        )
        for name, component in subject_components.items():
            plot_colored_graph(
                component,
                seed=16,
                title=f"Subject {subject}: {name}",
                output=arguments.output_dir / f"subject-{subject}-{name}.png",
            )
        print(
            f"[{_timestamp()}] [fMRI] DONE  Subject {subject} "
            f"| elapsed={_duration(perf_counter() - subject_started)} "
            f"| overall={position}/2 ({50 * position:.1f}%)",
            flush=True,
        )
    summary["complete"] = True
    write_json(summary, checkpoint_path)
    print(
        f"[{_timestamp()}] [fMRI] COMPLETE 2/2 subjects",
        flush=True,
    )
    print(f"Wrote figures and models to {arguments.output_dir}")


if __name__ == "__main__":
    main()
