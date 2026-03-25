"""
Calculator module for computing statistics from evaluation data.

This module provides pure calculation functions that have no side effects
(no file I/O, no plotting). This separation makes the code easier to test
and reuse.
"""

from .errors import ErrorCalculator, ErrorStatistics
from .precision_recall import PrecisionRecallCalculator, PRStatistics, CurveData
from .accuracy import compute_split_accuracies, summarize_accuracies

__all__ = [
    "ErrorCalculator",
    "ErrorStatistics",
    "PrecisionRecallCalculator",
    "PRStatistics",
    "CurveData",
    "compute_split_accuracies",
    "summarize_accuracies",
]
