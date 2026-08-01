"""Python port of applications/airquality/airquality.r."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

PORT_ROOT = Path(__file__).resolve().parents[1]
# Trong kho mã nguồn gốc, mã Python nằm trong python-port/ nên dữ liệu ở thư mục
# cha; ở kho này mã nằm ngay gốc kho, nên hai đường dẫn trùng nhau.
REPO_ROOT = PORT_ROOT.parent if (PORT_ROOT.parent / "applications").is_dir() else PORT_ROOT
sys.path.insert(0, str(PORT_ROOT / "src"))

from backward_cgm_pd.io import (
    load_air_quality_residuals,
    load_saved_pdglasso,
    load_saved_search,
    read_json,
    write_json,
)
from backward_cgm_pd.pdglasso import select_pdglasso
from backward_cgm_pd.graph import ColoredGraph
from backward_cgm_pd.plotting import (
    plot_air_quality_matrix_figure,
    plot_colored_graph,
)
from backward_cgm_pd.rcon import ModelTester
from backward_cgm_pd.search_tau import backward_cgm_pd

CHECKPOINT_VERSION = 3


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT
        / "applications"
        / "airquality"
        / "airdata"
        / "airdataAR1.RData",
    )
    parser.add_argument("--grid-points", type=int, default=10)
    parser.add_argument("--skip-pdglasso", action="store_true")
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--itmax", type=int, default=1_000)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument(
        "--air-results",
        type=Path,
        default=REPO_ROOT
        / "applications"
        / "airquality"
        / "airresults",
        help="Original R artifacts used for paper-parity checks and Figure 9.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed stages from an existing output JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PORT_ROOT / "python-results" / "air-quality.json",
    )
    arguments = parser.parse_args()

    data = load_air_quality_residuals(arguments.data)
    n = data.shape[0]
    covariance = np.cov(data, rowvar=False, ddof=1)
    standard = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(standard, standard)
    settings = {
        "input": str(arguments.data),
        "grid_points": arguments.grid_points,
        "skip_pdglasso": arguments.skip_pdglasso,
        "skip_backward": arguments.skip_backward,
        "alpha": arguments.alpha,
        "itmax": arguments.itmax,
        "parallel": arguments.parallel,
        "air_results": str(arguments.air_results),
    }
    stages = []
    if not arguments.skip_pdglasso:
        stages.extend(("pdglasso_covariance", "pdglasso_correlation"))
    if not arguments.skip_backward:
        stages.append("backward")
    stage_positions = {name: index for index, name in enumerate(stages, 1)}
    output: dict[str, object] = {
        "n": n,
        "p": data.shape[1],
        "input": str(arguments.data),
        "sample_variances": np.diag(covariance).tolist(),
        "sample_correlation_diagonal": np.diag(correlation).tolist(),
        "checkpoint_version": CHECKPOINT_VERSION,
        "settings": settings,
        "complete": False,
    }
    if arguments.resume and arguments.output.exists():
        candidate = read_json(arguments.output)
        if (
            candidate.get("checkpoint_version") == CHECKPOINT_VERSION
            and candidate.get("settings") == settings
        ):
            output = candidate
            print(f"Resuming checkpoint {arguments.output}")
        else:
            print("Ignoring incompatible checkpoint and starting a fresh run.")

    def checkpoint() -> None:
        output["complete"] = False
        write_json(output, arguments.output)
        completed = sum(stage in output for stage in stages)
        percent = 100 * completed / len(stages) if stages else 100.0
        print(
            f"[{_timestamp()}] [AirQuality] CHECKPOINT "
            f"{completed}/{len(stages)} ({percent:.1f}%) "
            f"| {arguments.output}",
            flush=True,
        )

    checkpoint()
    figures_dir = arguments.output.with_suffix("")
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(np.diag(covariance), "o-")
    axis.set(
        xlabel="variable index",
        ylabel="sample variance",
        title="Air Quality variable scales",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures_dir / "variable-scales.png", dpi=180)
    plt.close(figure)

    if not arguments.skip_pdglasso:
        for name, matrix in (("covariance", covariance), ("correlation", correlation)):
            output_key = f"pdglasso_{name}"
            if output_key in output:
                print(
                    f"[{_timestamp()}] [AirQuality] RESUME skip "
                    f"{output_key} ({stage_positions[output_key]}/{len(stages)})",
                    flush=True,
                )
                continue
            print(
                f"[{_timestamp()}] [AirQuality] START {output_key} "
                f"({stage_positions[output_key]}/{len(stages)}) "
                f"| lambda-grid={arguments.grid_points}",
                flush=True,
            )
            stage_started = perf_counter()
            selected = select_pdglasso(
                matrix,
                n,
                gamma_ebic=0.0,
                points=arguments.grid_points,
                verbose=arguments.verbose,
            )
            selection_runtime = perf_counter() - stage_started
            # The article tests every selected graph against the saturated
            # likelihood computed from the unscaled residual data.
            tester = ModelTester(data)
            _, pvalue = tester.test(selected.graph)
            selected_path_row = min(selected.lambda2_path, key=lambda row: row[1])
            output[output_key] = {
                "model": selected.graph.to_dict(),
                "selected_precision": selected.model.precision,
                "best_lambdas": selected.best_lambdas,
                "lambda_grid": {
                    "lambda1": [row[0] for row in selected.lambda1_path],
                    "lambda2": [row[0] for row in selected.lambda2_path],
                },
                "selected_eBIC": selected_path_row[1],
                "selected_log_likelihood_per_observation": selected_path_row[2],
                "selected_number_parameters": selected_path_row[3],
                "lrt_pvalue_on_unscaled_data": pvalue,
                "optimizer": {
                    "converged": selected.model.converged,
                    "iterations": selected.model.iterations,
                    "primal_residual": selected.model.primal_residual,
                    "dual_residual": selected.model.dual_residual,
                    "execution_time_seconds": selected.model.execution_time,
                },
                "total_selection_runtime_seconds": selection_runtime,
                "path_columns": [
                    "penalty",
                    "eBIC",
                    "log_likelihood_per_observation",
                    "number_parameters",
                    "converged",
                ],
                "lambda1_path": selected.lambda1_path,
                "lambda2_path": selected.lambda2_path,
            }
            axis = plot_colored_graph(
                selected.graph,
                title=f"pdglasso on {name}",
                output=figures_dir / f"pdglasso-{name}.png",
            )
            plt.close(axis.figure)
            print(
                f"[{_timestamp()}] [AirQuality] DONE  {output_key} "
                f"| elapsed={_duration(perf_counter() - stage_started)}",
                flush=True,
            )
            checkpoint()

    if not arguments.skip_backward:
        if "backward" in output:
            print(
                f"[{_timestamp()}] [AirQuality] RESUME skip backward "
                f"({stage_positions['backward']}/{len(stages)})",
                flush=True,
            )
        else:
            print(
                f"[{_timestamp()}] [AirQuality] START backward "
                f"({stage_positions['backward']}/{len(stages)}) "
                f"| itmax={arguments.itmax}",
                flush=True,
            )
            stage_started = perf_counter()
            result = backward_cgm_pd(
                data,
                itmax=arguments.itmax,
                alpha=arguments.alpha,
                optimizer_maxiter=500,
                tolerance=1e-7,
                n_jobs=arguments.parallel,
                rcon_backend="grc_ipms",
                count_initial_models=False,
                verbose=arguments.verbose,
            )
            output["backward"] = result.to_dict()
            output["backward"]["runtime_seconds"] = (
                perf_counter() - stage_started
            )
            output["backward"]["selected_log_likelihood"] = (
                result.fit.log_likelihood
            )
            output["backward"]["selected_number_parameters"] = (
                result.model.n_parameters
            )
            output["backward"]["lrt_pvalue_on_unscaled_data"] = result.pvalue
            axis = plot_colored_graph(
                result.model,
                title="Backward twin-lattice model",
                output=figures_dir / "backward.png",
            )
            plt.close(axis.figure)
            print(
                f"[{_timestamp()}] [AirQuality] DONE  backward "
                f"| elapsed={_duration(perf_counter() - stage_started)}",
                flush=True,
            )
            checkpoint()

    output["complete"] = True
    required_models = {
        "backward",
        "pdglasso_covariance",
        "pdglasso_correlation",
    }
    if required_models <= output.keys():
        def graph_from_output(key: str) -> ColoredGraph:
            model = output[key]["model"]
            return ColoredGraph.from_r(
                int(model["p"]),
                model["L.as"],
                model["E"],
                model["E.as"],
            )

        article_figure = plot_air_quality_matrix_figure(
            graph_from_output("backward"),
            graph_from_output("pdglasso_covariance"),
            graph_from_output("pdglasso_correlation"),
            output=figures_dir / "figure-9-python-refit.png",
        )
        plt.close(article_figure)
        print(
            f"[{_timestamp()}] [AirQuality] PYTHON REFIT FIGURE "
            f"| {figures_dir / 'figure-9-python-refit.png'}",
            flush=True,
        )

    backward_artifact = arguments.air_results / "air.backward.RData"
    covariance_artifact = arguments.air_results / "air.pdglassoS.RData"
    correlation_artifact = arguments.air_results / "air.pdglassoP.RData"
    if all(
        path.exists()
        for path in (
            backward_artifact,
            covariance_artifact,
            correlation_artifact,
        )
    ):
        reference_backward, backward_metadata = load_saved_search(
            backward_artifact,
            p=data.shape[1],
        )
        reference_covariance, covariance_metadata = load_saved_pdglasso(
            covariance_artifact
        )
        reference_correlation, correlation_metadata = load_saved_pdglasso(
            correlation_artifact
        )
        paper_figure = plot_air_quality_matrix_figure(
            reference_backward,
            reference_covariance,
            reference_correlation,
            output=figures_dir / "figure-9-matrix-models.png",
        )
        plt.close(paper_figure)

        reference = {
            "source": "original R artifacts supplied by the repository",
            "backward": {
                "model": reference_backward.to_dict(),
                "iterations": int(
                    np.asarray(backward_metadata["iterations"]).reshape(-1)[0]
                ),
                "no.models": int(
                    np.asarray(backward_metadata["no.models"]).reshape(-1)[0]
                ),
            },
            "pdglasso_covariance": {
                "model": reference_covariance.to_dict(),
                "best_lambdas": covariance_metadata["best_lambdas"],
            },
            "pdglasso_correlation": {
                "model": reference_correlation.to_dict(),
                "best_lambdas": correlation_metadata["best_lambdas"],
            },
        }
        parity = {}
        for key in (
            "backward",
            "pdglasso_covariance",
            "pdglasso_correlation",
        ):
            if key in output:
                parity[key] = {
                    "selected_graph_matches_r_artifact": (
                        output[key]["model"] == reference[key]["model"]
                    )
                }
        if "backward" in output:
            parity["backward"].update(
                {
                    "iterations_match": (
                        int(output["backward"]["iterations"])
                        == reference["backward"]["iterations"]
                    ),
                    "number_models_match": (
                        int(output["backward"]["no.models"])
                        == reference["backward"]["no.models"]
                    ),
                }
            )
        for key in ("pdglasso_covariance", "pdglasso_correlation"):
            if key in output:
                parity[key]["best_lambdas_match"] = bool(
                    np.allclose(
                        output[key]["best_lambdas"],
                        reference[key]["best_lambdas"],
                        rtol=1e-10,
                        atol=1e-12,
                    )
                )
        output["r_artifact_reference"] = reference
        output["parity_with_r_artifacts"] = parity
        print(
            f"[{_timestamp()}] [AirQuality] PAPER FIGURE 9 "
            f"| {figures_dir / 'figure-9-matrix-models.png'}",
            flush=True,
        )

    write_json(output, arguments.output)
    print(
        f"[{_timestamp()}] [AirQuality] COMPLETE {len(stages)}/{len(stages)} stages",
        flush=True,
    )
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
