"""
Plotter module for creating visualizations.

This module provides plotter classes that create matplotlib figures.
Key design principle: plotters return figures but don't save them.
Saving is handled by FileSaver at the pipeline level.
"""

from .base import BasePlotter
from .error_rate import ErrorRatePlotter
from .precision_recall import PrecisionRecallPlotter
from .dimensionality import DimensionalityPlotter
from .distribution import DistributionPlotter
from .accuracy import AccuracyPlotter
from .confusion_matrix import ConfusionMatrixPlotter

__all__ = [
    "BasePlotter",
    "ErrorRatePlotter",
    "PrecisionRecallPlotter",
    "DimensionalityPlotter",
    "DistributionPlotter",
    "AccuracyPlotter",
    "ConfusionMatrixPlotter",
]
