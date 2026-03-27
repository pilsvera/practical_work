"""
Visualization package for thesis evaluation results.

This package provides a modular, testable, and maintainable system for
generating visualizations from model evaluation results.

Quick Start:
    from masterthesis.visualization import VisualizationConfig, VisualizationPipeline
    
    config = VisualizationConfig.from_yaml("configs/source_3.yaml")
    pipeline = VisualizationPipeline(config)
    pipeline.run(plots=["error_rate", "accuracy"])

Package Structure:
    - config: Configuration dataclasses and YAML loading
    - constants: Color palettes and column definitions
    - io: File loading and saving utilities
    - calculators: Pure statistics computation (no side effects)
    - plotters: Figure generation (returns figures, doesn't save)
    - pipeline: Orchestration layer
    - cli: Command-line interface

Key Design Principles:
    1. Separation of concerns: loading, calculating, plotting, saving
    2. Testability: pure functions, dependency injection
    3. Configurability: YAML configs, sensible defaults
    4. Reusability: modular components that work independently
"""

from .config import VisualizationConfig, PlotSettings, SourceMappings, ExperimentPaths
from .pipeline import VisualizationPipeline, PLOT_TYPES
from .constants import (
    DEFAULT_COLUMNS,
    SPLIT_COLORS,
    LABEL_COLORS,
    get_label_colormap,
)

# Calculators
from .calculators import (
    ErrorCalculator,
    ErrorStatistics,
    PrecisionRecallCalculator,
    PRStatistics,
    CurveData,
    compute_split_accuracies,
    summarize_accuracies,
)

# Plotters
from .plotters import (
    BasePlotter,
    ErrorRatePlotter,
    PrecisionRecallPlotter,
    DimensionalityPlotter,
    DistributionPlotter,
    AccuracyPlotter,
)

# I/O
from .io import (
    NPYLoader,
    JSONLoader,
    EvaluationDataLoader,
    make_dataframe,
    FileSaver,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "VisualizationConfig",
    "PlotSettings",
    "SourceMappings",
    "ExperimentPaths",
    # Pipeline
    "VisualizationPipeline",
    "PLOT_TYPES",
    # Constants
    "DEFAULT_COLUMNS",
    "SPLIT_COLORS",
    "LABEL_COLORS",
    "get_label_colormap",
    # Calculators
    "ErrorCalculator",
    "ErrorStatistics",
    "PrecisionRecallCalculator",
    "PRStatistics",
    "CurveData",
    "compute_split_accuracies",
    "summarize_accuracies",
    # Plotters
    "BasePlotter",
    "ErrorRatePlotter",
    "PrecisionRecallPlotter",
    "DimensionalityPlotter",
    "DistributionPlotter",
    "AccuracyPlotter",
    # I/O
    "NPYLoader",
    "JSONLoader",
    "EvaluationDataLoader",
    "make_dataframe",
    "FileSaver",
]
