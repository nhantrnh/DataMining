"""Python port of the pdglasso ADMM used by the Air Quality experiment.

Only the default VIA model class used by ``airquality.r`` is exposed.  The
double ADMM follows the pure-R reference implementation from the open-source
``pdglasso`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .graph import ColoredGraph, full_edges, out_edges
from .rcon import fit_rcon

FloatArray = NDArray[np.float64]


def _upper_pairs(q: int) -> list[tuple[int, int]]:
    # R's upper.tri vector order (column-major).
    return [(i, j) for j in range(1, q) for i in range(j)]


def matrix_to_vector(matrix: ArrayLike) -> FloatArray:
    value = np.asarray(matrix, dtype=float)
    p = value.shape[0]
    q = p // 2
    pairs = _upper_pairs(q)
    return np.asarray(
        [value[i, i] for i in range(q)]
        + [value[q + i, q + i] for i in range(q)]
        + [value[i, j] for i, j in pairs]
        + [value[q + i, q + j] for i, j in pairs]
        + [value[i, q + j] for i, j in pairs]
        + [value[q + i, j] for i, j in pairs]
        + [value[i, q + i] for i in range(q)],
        dtype=float,
    )


def vector_to_matrix(vector: ArrayLike) -> FloatArray:
    value = np.asarray(vector, dtype=float)
    p = int((-1 + np.sqrt(1 + 8 * len(value))) / 2)
    q = p // 2
    pairs = _upper_pairs(q)
    h = len(pairs)
    matrix = np.zeros((p, p), dtype=float)
    offset = 0
    for i in range(q):
        matrix[i, i] = value[offset + i]
    offset += q
    for i in range(q):
        matrix[q + i, q + i] = value[offset + i]
    offset += q
    for block in range(4):
        segment = value[offset : offset + h]
        for number, (i, j) in enumerate(pairs):
            if block == 0:
                row, column = i, j
            elif block == 1:
                row, column = q + i, q + j
            elif block == 2:
                row, column = i, q + j
            else:
                # t(inv_half_vec(segment)) inside the LR block.
                row, column = j, q + i
            matrix[row, column] = segment[number]
        offset += h
    for i in range(q):
        matrix[i, q + i] = value[offset + i]
    matrix = matrix + np.triu(matrix, 1).T
    return matrix


def _difference_matrix(p: int) -> FloatArray:
    """F for vertex, inside-block and across-block twin differences."""
    q = p // 2
    h = q * (q - 1) // 2
    d = p * (p + 1) // 2
    F = np.zeros((q + 2 * h, d), dtype=float)
    row = 0
    blocks = ((0, q, q), (2 * q, 2 * q + h, h), (2 * q + 2 * h, 2 * q + 3 * h, h))
    for first, second, size in blocks:
        for index in range(size):
            F[row, first + index] = 1.0
            F[row, second + index] = -1.0
            row += 1
    return F


def _soft_threshold(value: FloatArray, threshold: ArrayLike) -> FloatArray:
    threshold_array = np.asarray(threshold, dtype=float)
    return np.sign(value) * np.maximum(np.abs(value) - threshold_array, 0.0)


@dataclass(frozen=True)
class PDGlassoFit:
    precision: FloatArray
    lambda1: float
    lambda2: float
    converged: bool
    iterations: int
    primal_residual: float
    dual_residual: float
    execution_time: float
    eps_relative: float


def fit_pdglasso(
    covariance: ArrayLike,
    lambda1: float,
    lambda2: float,
    *,
    rho1: float = 1.0,
    rho2: float = 1.0,
    varying_rho1: bool = True,
    varying_rho2: bool = True,
    maxiter: int = 5_000,
    eps_absolute: float = 1e-6,
    eps_relative: float = 1e-6,
    initial_precision: ArrayLike | None = None,
) -> PDGlassoFit:
    S = np.asarray(covariance, dtype=float)
    p = S.shape[0]
    if S.shape != (p, p) or p % 2:
        raise ValueError("covariance must be a square matrix of even order")
    F = _difference_matrix(p)
    d = p * (p + 1) // 2
    X = np.eye(p) if initial_precision is None else np.asarray(initial_precision, dtype=float)
    U = np.zeros((p, p))
    Z = np.zeros((p, p))
    start = perf_counter()
    primal = dual = np.inf

    def inner(current_X: FloatArray, current_U: FloatArray, outer_rho: float) -> FloatArray:
        b = matrix_to_vector(current_X + current_U)
        v = np.zeros(F.shape[0])
        t = np.zeros(F.shape[0])
        local_rho = rho2
        x = b.copy()
        for _ in range(maxiter):
            alpha = local_rho / (1.0 + 2.0 * local_rho)
            x = alpha * F.T @ (v - t - F @ b) + b
            Fx = F @ x
            last_v = v.copy()
            v = _soft_threshold(Fx + t, lambda2 / outer_rho / local_rho)
            t = Fx + t - v
            residual = float(np.linalg.norm(Fx - v))
            dual_residual = float(np.linalg.norm(local_rho * F.T @ (v - last_v)))
            eps_primal = np.sqrt(F.shape[0]) * eps_absolute + eps_relative * max(
                np.linalg.norm(Fx), np.linalg.norm(v)
            )
            eps_dual = np.sqrt(d) * eps_absolute + eps_relative * np.linalg.norm(
                F.T @ (local_rho * t)
            )
            if residual < eps_primal and dual_residual < eps_dual:
                break
            if varying_rho2:
                if residual > 10 * dual_residual:
                    local_rho *= 2
                    t /= 2
                elif dual_residual > 10 * residual:
                    local_rho /= 2
                    t *= 2
        return vector_to_matrix(_soft_threshold(x, lambda1 / outer_rho))

    converged = False
    eps_primal = eps_dual = np.inf
    for iteration in range(1, maxiter + 1):
        A = rho1 * (Z - U) - S
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        transformed = (
            eigenvalues + np.sqrt(eigenvalues**2 + 4.0 * rho1)
        ) / (2.0 * rho1)
        X = (eigenvectors * transformed) @ eigenvectors.T
        previous_Z = Z.copy()
        Z = inner(X, U, rho1)
        U += X - Z
        upper = np.triu_indices(p)
        primal = float(np.linalg.norm((X - Z)[upper]))
        dual = float(rho1 * np.linalg.norm((Z - previous_Z)[upper]))
        eps_primal = np.sqrt(d) * eps_absolute + eps_relative * max(
            np.linalg.norm(X), np.linalg.norm(Z)
        )
        eps_dual = np.sqrt(d) * eps_absolute + eps_relative * rho1 * np.linalg.norm(U)
        if primal < eps_primal and dual < eps_dual:
            converged = True
            break
        if varying_rho1:
            scale = rho1
            if primal > 10 * dual / scale:
                rho1 *= 2
                U /= 2
            elif dual / scale > 10 * primal:
                rho1 /= 2
                U *= 2
    return PDGlassoFit(
        precision=X,
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        converged=converged,
        iterations=iteration,
        primal_residual=primal,
        dual_residual=dual,
        execution_time=perf_counter() - start,
        eps_relative=eps_relative,
    )


def graph_from_pdglasso(
    fit: PDGlassoFit,
    *,
    zero_threshold: float | None = None,
    symmetry_threshold: float | None = None,
) -> ColoredGraph:
    X = fit.precision
    p = X.shape[0]
    q = p // 2
    # pdColG.get() in pdglasso 1.0.0 uses eps.rel * 100 for both defaults.
    threshold1 = (
        fit.eps_relative * 100 if zero_threshold is None else zero_threshold
    )
    threshold2 = (
        fit.eps_relative * 100
        if symmetry_threshold is None
        else symmetry_threshold
    )
    present = np.abs(X) > threshold1
    np.fill_diagonal(present, True)
    symmetric = np.zeros((p, p), dtype=bool)
    for i in range(q):
        if abs(X[i, i] - X[q + i, q + i]) <= threshold2:
            symmetric[i, i] = symmetric[q + i, q + i] = True
    for i, j in _upper_pairs(q):
        if (
            present[i, j] or present[q + i, q + j]
        ) and abs(X[i, j] - X[q + i, q + j]) <= threshold2:
            present[i, j] = present[j, i] = True
            present[q + i, q + j] = present[q + j, q + i] = True
            symmetric[i, j] = symmetric[j, i] = True
            symmetric[q + i, q + j] = symmetric[q + j, q + i] = True
        if (
            present[i, q + j] or present[j, q + i]
        ) and abs(X[i, q + j] - X[j, q + i]) <= threshold2:
            present[i, q + j] = present[q + j, i] = True
            present[j, q + i] = present[q + i, j] = True
            symmetric[i, q + j] = symmetric[q + j, i] = True
            symmetric[j, q + i] = symmetric[q + i, j] = True
    left_atomic = frozenset(i + 1 for i in range(q) if not symmetric[i, i])
    edges = frozenset(
        (i + 1, j + 1)
        for i in range(p)
        for j in range(i + 1, p)
        if present[i, j]
    )
    candidates = out_edges(edges, p).TE
    atomic = frozenset(
        edge
        for edge in candidates
        if not symmetric[edge[0] - 1, edge[1] - 1]
    )
    return ColoredGraph(p, left_atomic, edges, atomic)


def lambda_maxima(covariance: ArrayLike) -> tuple[float, float]:
    S = np.asarray(covariance, dtype=float)
    p = S.shape[0]
    q = p // 2
    off_diagonal = np.abs(S[np.triu_indices(p, 1)])
    inside = np.abs(S[:q, :q] - S[q:, q:]) / 2
    across = np.abs(S[:q, q:] - S[:q, q:].T) / 2
    return float(np.max(off_diagonal)), float(max(np.max(inside), np.max(across)))


@dataclass(frozen=True)
class PDGlassoSelection:
    model: PDGlassoFit
    graph: ColoredGraph
    best_lambdas: tuple[float, float]
    lambda1_path: tuple[tuple[float, float, float, int, bool], ...]
    lambda2_path: tuple[tuple[float, float, float, int, bool], ...]


def _ebic(
    covariance: FloatArray,
    n: int,
    graph: ColoredGraph,
    gamma: float,
) -> tuple[float, float, int]:
    # pdglasso::compute.eBIC first obtains the MLE using the unadjusted
    # sample covariance S, and only then evaluates log|K|-tr((n-1)/n*S*K).
    mle = fit_rcon(None, graph, covariance=covariance, n_observations=n)
    adjusted = covariance * (n - 1) / n
    sign, logdet = np.linalg.slogdet(mle.precision)
    if sign <= 0:
        raise RuntimeError("pdRCON MLE is not positive definite")
    per_observation = float(
        logdet - np.sum(adjusted * mle.precision)
    )
    parameters = graph.n_parameters
    value = (
        -n * per_observation
        + np.log(n) * parameters
        + 4 * parameters * gamma * np.log(graph.p)
    )
    return float(value), float(per_observation), parameters


def select_pdglasso(
    covariance: ArrayLike,
    n: int,
    *,
    gamma_ebic: float = 0.0,
    points: int = 10,
    maxiter: int = 5_000,
    eps_absolute: float = 1e-8,
    eps_relative: float = 1e-8,
    verbose: bool = False,
) -> PDGlassoSelection:
    S = np.asarray(covariance, dtype=float)
    max1, max2 = lambda_maxima(S)
    # pdRCON.fit 1.0.0 defaults: linear grids from max/10 to max, sorted
    # decreasing.  These produce 10 lambda1 and 11 lambda2 path rows.
    grid1 = np.linspace(max1 / 10, max1, points)[::-1]
    grid2 = np.linspace(max2 / 10, max2, points)[::-1]

    path1: list[tuple[float, float, float, int, bool]] = []
    fits1: list[PDGlassoFit] = []
    graphs1: list[ColoredGraph] = []
    for index, penalty in enumerate(grid1, 1):
        if verbose:
            print(f"[pdglasso] lambda1 {index}/{points}", flush=True)
        fit = fit_pdglasso(
            S,
            float(penalty),
            0.0,
            maxiter=maxiter,
            eps_absolute=eps_absolute,
            eps_relative=eps_relative,
        )
        graph = graph_from_pdglasso(fit)
        ebic, loglik, parameters = _ebic(S, n, graph, gamma_ebic)
        path1.append(
            (float(penalty), ebic, loglik, parameters, fit.converged)
        )
        fits1.append(fit)
        graphs1.append(graph)
    best1_index = int(np.argmin([row[1] for row in path1]))
    best1 = float(grid1[best1_index])

    path2: list[tuple[float, float, float, int, bool]] = []
    fits2: list[PDGlassoFit] = []
    graphs2: list[ColoredGraph] = []
    for index, penalty in enumerate(grid2, 1):
        if verbose:
            print(f"[pdglasso] lambda2 {index}/{points}", flush=True)
        fit = fit_pdglasso(
            S,
            best1,
            float(penalty),
            maxiter=maxiter,
            eps_absolute=eps_absolute,
            eps_relative=eps_relative,
        )
        graph = graph_from_pdglasso(fit)
        ebic, loglik, parameters = _ebic(S, n, graph, gamma_ebic)
        path2.append(
            (float(penalty), ebic, loglik, parameters, fit.converged)
        )
        fits2.append(fit)
        graphs2.append(graph)
    # lambda2=0 candidate is inherited from the first grid.
    path2.append((0.0, *path1[best1_index][1:]))
    fits2.append(fits1[best1_index])
    graphs2.append(graphs1[best1_index])
    best2_index = int(np.argmin([row[1] for row in path2]))
    return PDGlassoSelection(
        model=fits2[best2_index],
        graph=graphs2[best2_index],
        best_lambdas=(best1, path2[best2_index][0]),
        lambda1_path=tuple(path1),
        lambda2_path=tuple(path2),
    )
