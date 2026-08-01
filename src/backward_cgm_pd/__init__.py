"""Coloured Gaussian graphical models for paired data."""

from .graph import ColoredGraph, full_edges, out_edges, tau, tau_edge
from .rcon import RCONFit, fit_rcon, likelihood_ratio_pvalue
from .search_tau import SearchResult, backward_cgm_pd
from .search_submodel import backward_submodel

__all__ = [
    "ColoredGraph",
    "RCONFit",
    "SearchResult",
    "backward_cgm_pd",
    "backward_submodel",
    "fit_rcon",
    "full_edges",
    "likelihood_ratio_pvalue",
    "out_edges",
    "tau",
    "tau_edge",
]
