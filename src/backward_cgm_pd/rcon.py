"""Maximum-likelihood fitting and LRTs for RCON models.

The original R implementation delegates this step to ``gRc::rcox(method="ipms")``.
This port optimises the same Gaussian likelihood directly over one free
parameter per vertex/edge colour class.  The feasible set is convex; invalid
(non positive-definite) trial matrices are kept outside the log-det barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2
from joblib import Parallel, delayed

from .graph import ColoredGraph, graph_key

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class RCONFit:
    graph: ColoredGraph
    precision: FloatMatrix
    covariance: FloatMatrix
    log_likelihood: float
    n_parameters: int
    converged: bool
    message: str
    iterations: int


def _positive_logdet(matrix: FloatMatrix) -> float | None:
    """Return log(det(matrix)) only when matrix is positive definite."""
    try:
        cholesky = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return None
    return float(2.0 * np.sum(np.log(np.diag(cholesky))))


def _basis_matrices(graph: ColoredGraph) -> tuple[FloatMatrix, ...]:
    p = graph.p
    bases: list[FloatMatrix] = []
    for colour_class in graph.vertex_colour_classes():
        basis = np.zeros((p, p), dtype=float)
        for vertex in colour_class:
            basis[vertex - 1, vertex - 1] = 1.0
        bases.append(basis)
    for colour_class in graph.edge_colour_classes():
        basis = np.zeros((p, p), dtype=float)
        for i, j in colour_class:
            basis[i - 1, j - 1] = basis[j - 1, i - 1] = 1.0
        bases.append(basis)
    return tuple(bases)


def _theta_from_precision(graph: ColoredGraph, precision: FloatMatrix) -> NDArray[np.float64]:
    values: list[float] = []
    for colour_class in graph.vertex_colour_classes():
        values.append(float(np.mean([precision[v - 1, v - 1] for v in colour_class])))
    for colour_class in graph.edge_colour_classes():
        values.append(float(np.mean([precision[i - 1, j - 1] for i, j in colour_class])))
    return np.asarray(values)


def _matrix_from_theta(theta: NDArray[np.float64], bases: tuple[FloatMatrix, ...]) -> FloatMatrix:
    matrix = np.zeros_like(bases[0])
    for value, basis in zip(theta, bases, strict=True):
        matrix += value * basis
    return matrix


def _conjugate_gradient(
    hessian_product,
    right_hand_side: NDArray[np.float64],
    *,
    relative_tolerance: float = 1e-8,
) -> NDArray[np.float64]:
    """Solve a positive-definite Newton system using matrix-free CG."""
    solution = np.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    direction = residual.copy()
    squared = float(residual @ residual)
    initial = max(squared, np.finfo(float).tiny)
    for _ in range(max(20, 2 * len(right_hand_side))):
        product = hessian_product(direction)
        curvature = float(direction @ product)
        if curvature <= np.finfo(float).tiny:
            break
        step = squared / curvature
        solution += step * direction
        residual -= step * product
        next_squared = float(residual @ residual)
        if next_squared <= initial * relative_tolerance**2:
            break
        direction = residual + (next_squared / squared) * direction
        squared = next_squared
    return solution


def fit_rcon(
    data: ArrayLike | None,
    graph: ColoredGraph,
    *,
    covariance: ArrayLike | None = None,
    n_observations: int | None = None,
    start_precision: ArrayLike | None = None,
    maxiter: int = 2_000,
    tolerance: float = 1e-9,
) -> RCONFit:
    """Fit an RCON model from raw data or an empirical covariance matrix."""
    if covariance is None:
        if data is None:
            raise ValueError("Provide data or covariance")
        values = np.asarray(data, dtype=float)
        if values.ndim != 2 or values.shape[1] != graph.p:
            raise ValueError(f"data must have shape (n, {graph.p})")
        covariance_matrix = np.cov(values, rowvar=False, ddof=1)
        n = values.shape[0]
    else:
        covariance_matrix = np.asarray(covariance, dtype=float)
        if covariance_matrix.shape != (graph.p, graph.p):
            raise ValueError(f"covariance must have shape ({graph.p}, {graph.p})")
        if n_observations is None:
            raise ValueError("n_observations is required with covariance")
        n = int(n_observations)

    covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2.0
    if start_precision is None:
        ridge = max(1e-10, 1e-8 * float(np.trace(covariance_matrix)) / graph.p)
        initial_precision = np.linalg.inv(
            covariance_matrix + ridge * np.eye(graph.p)
        )
    else:
        initial_precision = np.asarray(start_precision, dtype=float)

    bases = _basis_matrices(graph)
    theta0 = _theta_from_precision(graph, initial_precision)
    # Projecting a saturated estimate onto equality constraints need not remain
    # PD. Identity is always a valid starting point for every graph.
    diagonal = np.zeros((graph.p, graph.p), dtype=float)
    for colour_class in graph.vertex_colour_classes():
        indices = [vertex - 1 for vertex in colour_class]
        value = len(indices) / float(
            sum(covariance_matrix[index, index] for index in indices)
        )
        for index in indices:
            diagonal[index, index] = value

    projected = _matrix_from_theta(theta0, bases)
    if _positive_logdet(projected) is None:
        # Build a scale-aware feasible diagonal point.  This matters for the
        # unscaled Air Quality data, whose variances differ by several orders
        # of magnitude.  Blend toward it until the projected saturated
        # estimate enters the positive-definite cone.
        feasible_theta = _theta_from_precision(graph, diagonal)
        weight = 0.5
        while weight > 1e-8:
            trial_theta = weight * theta0 + (1.0 - weight) * feasible_theta
            if (
                _positive_logdet(_matrix_from_theta(trial_theta, bases))
                is not None
            ):
                theta0 = trial_theta
                break
            weight *= 0.5
        else:
            theta0 = feasible_theta

    # Optimise dimensionless coordinates. Without this preconditioner the Air
    # Quality covariance (very different variable scales) makes L-BFGS spend
    # hundreds of iterations on a few two-edge candidates.
    scales: list[float] = []
    for colour_class in graph.vertex_colour_classes():
        scales.append(
            float(np.mean([diagonal[v - 1, v - 1] for v in colour_class]))
        )
    for colour_class in graph.edge_colour_classes():
        scales.append(
            float(
                np.mean(
                    [
                        np.sqrt(diagonal[i - 1, i - 1] * diagonal[j - 1, j - 1])
                        for i, j in colour_class
                    ]
                )
            )
        )
    variance_diagonal = np.diag(covariance_matrix)
    scale_ratio = float(np.max(variance_diagonal) / np.min(variance_diagonal))
    parameter_scales = (
        np.maximum(np.asarray(scales), 1e-12)
        if scale_ratio > 1e3
        else np.ones(len(scales), dtype=float)
    )
    phi = theta0 / parameter_scales
    scaled_bases = np.asarray(
        [
            scale * basis
            for scale, basis in zip(parameter_scales, bases, strict=True)
        ]
    )

    converged = False
    message = "maximum iterations reached"
    iteration = 0
    previous_value = np.inf
    for iteration in range(1, maxiter + 1):
        precision = np.tensordot(phi, scaled_bases, axes=1)
        logdet = _positive_logdet(precision)
        if logdet is None:
            raise RuntimeError("RCON iteration left the positive-definite cone")
        inverse = np.linalg.inv(precision)
        value = -logdet + float(np.sum(covariance_matrix * precision))
        residual = covariance_matrix - inverse
        gradient = np.einsum("ij,aij->a", residual, scaled_bases)
        if float(np.max(np.abs(gradient))) <= tolerance:
            converged = True
            message = "gradient tolerance reached"
            break

        def hessian_product(vector: NDArray[np.float64]) -> NDArray[np.float64]:
            direction_matrix = np.tensordot(vector, scaled_bases, axes=1)
            transformed = inverse @ direction_matrix @ inverse
            return np.einsum("ij,aij->a", transformed, scaled_bases)

        newton_direction = _conjugate_gradient(hessian_product, -gradient)
        directional_derivative = float(gradient @ newton_direction)
        if directional_derivative >= 0:
            newton_direction = -gradient
            directional_derivative = -float(gradient @ gradient)

        step = 1.0
        accepted = False
        while step >= 2.0**-50:
            trial_phi = phi + step * newton_direction
            trial_precision = np.tensordot(trial_phi, scaled_bases, axes=1)
            trial_logdet = _positive_logdet(trial_precision)
            if trial_logdet is not None:
                trial_value = -trial_logdet + float(
                    np.sum(covariance_matrix * trial_precision)
                )
                if trial_value <= value + 1e-4 * step * directional_derivative:
                    phi = trial_phi
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            message = "positive-definite line search failed"
            break
        if (
            abs(previous_value - trial_value)
            <= tolerance * max(1.0, abs(value))
            and float(np.max(np.abs(gradient))) <= np.sqrt(tolerance)
        ):
            converged = True
            message = "objective and gradient tolerance reached"
            break
        previous_value = value

    precision = np.tensordot(phi, scaled_bases, axes=1)
    logdet = _positive_logdet(precision)
    if logdet is None:
        raise RuntimeError(f"RCON optimiser returned a non-PD matrix: {result.message}")
    fitted_covariance = np.linalg.inv(precision)
    # R receives cov(data), whose divisor is n-1; gRc evaluates the Gaussian
    # likelihood against the corresponding scatter matrix (n-1) * S.
    log_likelihood = 0.5 * (n - 1) * (
        logdet - float(np.sum(covariance_matrix * precision))
    )
    return RCONFit(
        graph=graph,
        precision=precision,
        covariance=fitted_covariance,
        log_likelihood=log_likelihood,
        n_parameters=len(bases),
        converged=converged,
        message=message,
        iterations=iteration,
    )


def _score_matching_start(
    covariance: FloatMatrix,
    graph: ColoredGraph,
    bases: tuple[FloatMatrix, ...],
) -> FloatMatrix:
    """Port of gRc::rconScoreMatch(), used as rcox's default Kstart."""
    basis_array = np.asarray(bases)
    count = len(basis_array)
    right = np.asarray([np.trace(basis) for basis in bases], dtype=float)
    system = np.einsum(
        "aij,jk,bki->ab",
        basis_array,
        covariance,
        basis_array,
        optimize=True,
    )
    try:
        theta = np.linalg.solve(system, right)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(system, right, rcond=None)[0]
    precision = _matrix_from_theta(theta, bases)

    # gRc repairs all diagonal entries if at least one is negative.
    if float(np.min(np.diag(precision))) < 0:
        covariance_diagonal = np.diag(covariance)
        for colour_class in graph.vertex_colour_classes():
            indices = [vertex - 1 for vertex in colour_class]
            precision[indices, indices] = 1.0 / float(
                np.mean(covariance_diagonal[indices])
            )

    # Port of gRc::regularizeK().
    if float(np.min(np.linalg.eigvalsh(precision))) < 0:
        diagonal = np.diag(np.abs(np.diag(precision)))
        remainder = precision - diagonal
        alpha = 0.9
        while alpha > 0:
            candidate = diagonal + alpha * remainder
            if float(np.min(np.linalg.eigvalsh(candidate))) > 0:
                precision = diagonal + 0.95 * alpha * remainder
                break
            alpha -= 0.1
        else:
            precision = diagonal
    return precision


def _generator_order(graph: ColoredGraph) -> tuple[FloatMatrix, ...]:
    """Match gRc IPM order: atomic then coloured VCCs, then ECCs."""
    vertex_classes = graph.vertex_colour_classes()
    edge_classes = graph.edge_colour_classes()
    ordered_classes = (
        tuple(item for item in vertex_classes if len(item) == 1)
        + tuple(item for item in vertex_classes if len(item) > 1)
        + tuple(item for item in edge_classes if len(item) == 1)
        + tuple(item for item in edge_classes if len(item) > 1)
    )
    bases: list[FloatMatrix] = []
    for colour_class in ordered_classes:
        basis = np.zeros((graph.p, graph.p), dtype=float)
        first = next(iter(colour_class))
        if isinstance(first, int):
            for vertex in colour_class:
                basis[vertex - 1, vertex - 1] = 1.0
        else:
            for i, j in colour_class:
                basis[i - 1, j - 1] = basis[j - 1, i - 1] = 1.0
        bases.append(basis)
    return tuple(bases)


def _ell_k(
    precision: FloatMatrix,
    covariance: FloatMatrix,
    n: int,
) -> float:
    logdet = _positive_logdet(precision)
    if logdet is None:
        return -np.inf
    return 0.5 * (n - 1) * (
        logdet - float(np.sum(covariance * precision))
    )


def fit_rcon_grc_ipms(
    data: ArrayLike | None,
    graph: ColoredGraph,
    *,
    covariance: ArrayLike | None = None,
    n_observations: int | None = None,
    maxouter: int = 500,
    maxinner: int = 10,
    log_likelihood_tolerance: float = 1e-6,
    delta_tolerance: float = 1e-3,
) -> RCONFit:
    """Numerical port of gRc::rcox(method="ipms") used by the R searches."""
    if covariance is None:
        if data is None:
            raise ValueError("Provide data or covariance")
        values = np.asarray(data, dtype=float)
        if values.ndim != 2 or values.shape[1] != graph.p:
            raise ValueError(f"data must have shape (n, {graph.p})")
        covariance_matrix = np.cov(values, rowvar=False, ddof=1)
        n = values.shape[0]
    else:
        covariance_matrix = np.asarray(covariance, dtype=float)
        if n_observations is None:
            raise ValueError("n_observations is required with covariance")
        n = int(n_observations)
    covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2.0
    parameter_bases = _basis_matrices(graph)
    precision = _score_matching_start(
        covariance_matrix,
        graph,
        parameter_bases,
    )
    generators = _generator_order(graph)
    previous_log_likelihood = _ell_k(precision, covariance_matrix, n)
    converged = False
    message = "maximum outer iterations reached"
    all_indices = np.arange(graph.p)
    generator_metadata = []
    for generator in generators:
        involved = np.flatnonzero(np.any(generator != 0, axis=0))
        complement = all_indices[
            ~np.isin(all_indices, involved, assume_unique=True)
        ]
        involved_block = np.ix_(involved, involved)
        complement_block = np.ix_(complement, complement)
        involved_complement = np.ix_(involved, complement)
        complement_involved = np.ix_(complement, involved)
        local_generator = generator[involved_block]
        local_covariance = covariance_matrix[involved_block]
        nonzero = np.nonzero(generator)
        generator_metadata.append(
            (
                involved_block,
                complement_block,
                involved_complement,
                complement_involved,
                local_generator,
                float(np.sum(local_generator.T * local_covariance)),
                nonzero,
            )
        )

    for outer_iteration in range(1, maxouter + 1):
        for (
            involved_block,
            complement_block,
            involved_complement,
            complement_involved,
            local_generator,
            target_trace,
            nonzero,
        ) in generator_metadata:
            if len(complement_block[0]):
                cross = precision[involved_complement]
                schur = cross @ np.linalg.solve(
                    precision[complement_block],
                    precision[complement_involved],
                )
            else:
                schur = 0.0
            previous_adjustment = 0.0
            for _ in range(maxinner):
                local_inverse = np.linalg.inv(
                    precision[involved_block] - schur
                )
                first_trace = float(
                    np.sum(local_generator.T * local_inverse)
                )
                second_trace = float(
                    np.trace(
                        local_generator
                        @ local_inverse
                        @ local_generator
                        @ local_inverse
                    )
                )
                delta = first_trace - target_trace
                adjustment = delta / (
                    second_trace + delta * delta / 2.0
                )
                precision[nonzero] += adjustment
                adjustment_change = adjustment - previous_adjustment
                previous_adjustment = adjustment
                if abs(adjustment_change) < delta_tolerance:
                    break

        log_likelihood = _ell_k(precision, covariance_matrix, n)
        if (
            log_likelihood - previous_log_likelihood
            < log_likelihood_tolerance
        ):
            converged = True
            message = "gRc IPM log-likelihood tolerance reached"
            break
        previous_log_likelihood = log_likelihood
    else:
        outer_iteration = maxouter
        log_likelihood = _ell_k(precision, covariance_matrix, n)

    return RCONFit(
        graph=graph,
        precision=precision,
        covariance=np.linalg.inv(precision),
        log_likelihood=log_likelihood,
        n_parameters=len(parameter_bases),
        converged=converged,
        message=message,
        iterations=outer_iteration,
    )


def likelihood_ratio_pvalue(candidate: RCONFit, saturated: RCONFit) -> float:
    degrees = saturated.n_parameters - candidate.n_parameters
    if degrees < 0:
        raise ValueError("candidate has more parameters than the reference model")
    if degrees == 0:
        return 1.0 if np.isclose(candidate.log_likelihood, saturated.log_likelihood) else 0.0
    statistic = max(
        0.0, -2.0 * (candidate.log_likelihood - saturated.log_likelihood)
    )
    return float(chi2.sf(statistic, degrees))


class ModelTester:
    """Fit/cache candidate models against one saturated reference."""

    def __init__(
        self,
        data: ArrayLike,
        *,
        maxiter: int = 2_000,
        tolerance: float = 1e-9,
        backend: str = "mle",
        n_jobs: int = 1,
    ) -> None:
        values = np.asarray(data, dtype=float)
        if values.ndim != 2 or values.shape[1] % 2:
            raise ValueError("paired data must be a 2D array with an even column count")
        self.data = values
        self.p = values.shape[1]
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.backend = backend
        self.n_jobs = max(1, int(n_jobs))
        self.saturated = self._fit(ColoredGraph.saturated(self.p))
        self._cache: dict[tuple[object, ...], tuple[RCONFit, float]] = {}

    def _fit(self, graph: ColoredGraph) -> RCONFit:
        if self.backend == "grc_ipms":
            return fit_rcon_grc_ipms(self.data, graph)
        if self.backend == "mle":
            return fit_rcon(
                self.data,
                graph,
                maxiter=self.maxiter,
                tolerance=self.tolerance,
            )
        raise ValueError(f"Unknown RCON backend: {self.backend}")

    def test(self, graph: ColoredGraph) -> tuple[RCONFit, float]:
        key = graph_key(graph)
        if key not in self._cache:
            fit = self._fit(graph)
            self._cache[key] = (fit, likelihood_ratio_pvalue(fit, self.saturated))
        return self._cache[key]

    def test_many(
        self,
        graphs: list[ColoredGraph],
    ) -> list[tuple[RCONFit, float]]:
        missing = [
            graph
            for graph in graphs
            if graph_key(graph) not in self._cache
        ]
        if missing:
            fits = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit)(graph) for graph in missing
            )
            for graph, fit in zip(missing, fits, strict=True):
                self._cache[graph_key(graph)] = (
                    fit,
                    likelihood_ratio_pvalue(fit, self.saturated),
                )
        return [self._cache[graph_key(graph)] for graph in graphs]

    @property
    def number_fitted(self) -> int:
        return len(self._cache)
