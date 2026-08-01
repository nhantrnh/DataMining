"""Matplotlib equivalents of ``layoutSym`` and ``outGraph``."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .graph import ColoredGraph, full_edges, out_edges, tau, tau_edge


def symmetric_layout(p: int, seed: int = 15) -> dict[int, tuple[float, float]]:
    q = p // 2
    if p == 4:
        return {1: (0, 2), 2: (0, 0), 3: (2, 2), 4: (2, 0)}
    middle = 6 if p < 40 else 8
    rng = np.random.default_rng(seed)
    left_x = 1.2 * rng.uniform(0, middle - 1.5, q)
    distance = np.abs(left_x - middle)
    positions: dict[int, tuple[float, float]] = {}
    for index in range(q):
        positions[index + 1] = (float(left_x[index]), -(index + 1) / 3)
        positions[q + index + 1] = (
            float(left_x[index] + 2 * distance[index]),
            -(index + 1) / 3,
        )
    return positions


def plot_colored_graph(
    graph: ColoredGraph,
    *,
    seed: int = 15,
    title: str | None = None,
    output: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    network = nx.Graph()
    network.add_nodes_from(range(1, graph.p + 1))
    network.add_edges_from(graph.E)
    q = graph.p // 2

    palette = plt.cm.rainbow(np.linspace(0, 1, max(1, q)))
    node_colors: list[object] = []
    for vertex in range(1, graph.p + 1):
        left = vertex if vertex <= q else tau(vertex, graph.p)
        node_colors.append("white" if left in graph.L_atomic else palette[left - 1])

    symmetric = out_edges(graph.E, graph.p).TE - graph.E_atomic
    symmetric_lookup = symmetric | {tau_edge(edge, graph.p) for edge in symmetric}
    edge_palette = plt.cm.rainbow(np.linspace(0, 1, max(1, len(symmetric))))
    colour_by_edge = {
        edge: edge_palette[index] for index, edge in enumerate(sorted(symmetric))
    }
    colour_by_edge.update(
        {tau_edge(edge, graph.p): colour for edge, colour in list(colour_by_edge.items())}
    )
    edge_colors = [
        colour_by_edge.get(tuple(sorted(edge)), "black") for edge in network.edges()
    ]
    labels = {
        vertex: f"L{vertex}" if vertex <= q else f"R{vertex - q}"
        for vertex in network.nodes
    }
    nx.draw_networkx(
        network,
        pos=symmetric_layout(graph.p, seed),
        labels=labels,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=500,
        width=2,
        edgecolors="black",
        ax=ax,
    )
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(destination, bbox_inches="tight", dpi=180)
    return ax


def plot_model_matrix(
    graph: ColoredGraph,
    *,
    ax: plt.Axes | None = None,
    panel_label: str | None = None,
) -> plt.Axes:
    """Plot the upper-triangular matrix representation used in Figure 9."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    p = graph.p
    q = p // 2
    twin_left = out_edges(graph.E, p).TE
    symmetric_left = twin_left - graph.E_atomic
    symmetric_full = symmetric_left | {
        tau_edge(edge, p) for edge in symmetric_left
    }
    transverse_edges = full_edges(p).FT

    grey: list[tuple[float, float]] = []
    black_circles: list[tuple[float, float]] = []
    black_squares: list[tuple[float, float]] = []

    # Diagonal entries represent vertex colour classes.
    for vertex in range(1, p + 1):
        left = vertex if vertex <= q else tau(vertex, p)
        target = (float(vertex), float(vertex))
        if left in graph.L_atomic:
            black_circles.append(target)
        else:
            grey.append(target)

    # Off-diagonal entries represent selected edges.
    for edge in sorted(graph.E):
        i, j = edge
        target = (float(j), float(i))
        if edge in symmetric_full:
            grey.append(target)
        elif edge in transverse_edges:
            black_squares.append(target)
        elif tau_edge(edge, p) in graph.E:
            black_circles.append(target)
        else:
            black_squares.append(target)

    def scatter(
        points: list[tuple[float, float]],
        *,
        marker: str,
        facecolor: str,
    ) -> None:
        if not points:
            return
        x, y = zip(*points)
        ax.scatter(
            x,
            y,
            s=48,
            marker=marker,
            facecolors=facecolor,
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )

    scatter(grey, marker="o", facecolor="#d9d9d9")
    scatter(black_circles, marker="o", facecolor="black")
    scatter(black_squares, marker="s", facecolor="black")

    divider = q + 0.5
    ax.axvline(divider, color="#555555", linewidth=0.8, zorder=1)
    ax.axhline(divider, color="#555555", linewidth=0.8, zorder=1)
    ax.set_xlim(0.5, p + 0.5)
    ax.set_ylim(p + 0.5, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    if panel_label:
        ax.text(
            0.5,
            -0.04,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )
    return ax


def plot_air_quality_matrix_figure(
    backward: ColoredGraph,
    unscaled: ColoredGraph,
    standardized: ColoredGraph,
    *,
    output: str | Path | None = None,
) -> plt.Figure:
    """Recreate the three-panel matrix representation in Figure 9."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for graph, axis, label in zip(
        (backward, unscaled, standardized),
        axes,
        ("(a)", "(b)", "(c)"),
        strict=True,
    ):
        plot_model_matrix(graph, ax=axis, panel_label=label)
    figure.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.10,
        wspace=0.14,
    )
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=220, bbox_inches="tight")
    return figure
