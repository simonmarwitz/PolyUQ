"""
polyuq — Polymorphic Uncertainty Quantification

Public API
----------
RandomVariable, MassFunction : uncertain variable types
PolyUQ                       : the propagation / analysis engine
DataManager                  : HPC result orchestration (optional heavy deps)

Usage
-----
from polyuq import RandomVariable, MassFunction, PolyUQ
"""

import logging

logger = logging.getLogger(__name__)

from polyuq.polymorphic_uncertainty import (
    RandomVariable,
    MassFunction,
    PolyUQ,
    compute_belief,
    aggregate_mass,
    plot_focals,
    stat_fun_avg,
    stat_fun_ci,
    stat_fun_lci,
    stat_fun_hist,
    stat_fun_pdf,
    stat_fun_cdf,
    weighted_quantile,
    generate_histogram_bins,
)

__all__ = [
    "RandomVariable",
    "MassFunction",
    "PolyUQ",
    "compute_belief",
    "aggregate_mass",
    "plot_focals",
    "stat_fun_avg",
    "stat_fun_ci",
    "stat_fun_lci",
    "stat_fun_hist",
    "stat_fun_pdf",
    "stat_fun_cdf",
    "weighted_quantile",
    "generate_histogram_bins",
]
