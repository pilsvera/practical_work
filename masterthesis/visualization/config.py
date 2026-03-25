"""
Configuration dataclasses and YAML loading for the visualization package.

This module provides type-safe configuration management using dataclasses
and supports loading configurations from YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

from .constants import (
    DEFAULT_FIGSIZE,
    DEFAULT_DPI,
    DEFAULT_LABEL_FONTSIZE,
    DEFAULT_TICK_FONTSIZE,
    DEFAULT_LEGEND_FONTSIZE,
)


@dataclass
class SourceMappings:
    """
    Maps column names to their encoding JSON files.
    
    These JSON files contain the mapping from encoded integers
    to human-readable labels (e.g., compound names, plate IDs).
    """
    labels: str
    plate_ids: str
    well_ids: str
    batch_ids: str
    
    def to_path_dict(self, base_dir: Path = None) -> Dict[str, Path]:
        """
        Convert string paths to Path objects.
        
        Args:
            base_dir: Optional base directory to prepend to paths
            
        Returns:
            Dictionary mapping column names to Path objects
        """
        paths = {
            "labels": Path(self.labels),
            "plate_ids": Path(self.plate_ids),
            "well_ids": Path(self.well_ids),
            "batch_ids": Path(self.batch_ids),
        }
        if base_dir:
            paths = {k: base_dir / v for k, v in paths.items()}
        return paths


@dataclass
class ExperimentPaths:
    """
    Checkpoint paths for a single split (e.g., batch/plate/well/random).
    
    Each split can have multiple replicates (runs) for statistical analysis.
    """
    split_name: str
    npy_paths: List[str]
    
    def get_paths(self, base_dir: Path = None) -> List[Path]:
        """
        Convert string paths to Path objects.
        
        Args:
            base_dir: Optional base directory to prepend to paths
            
        Returns:
            List of Path objects
        """
        paths = [Path(p) for p in self.npy_paths]
        if base_dir:
            paths = [base_dir / p for p in paths]
        return paths


@dataclass
class PlotSettings:
    """
    Visual settings for plots.
    
    Controls figure size, fonts, and display options.
    """
    figsize: tuple = DEFAULT_FIGSIZE
    dpi: int = DEFAULT_DPI
    label_fontsize: int = DEFAULT_LABEL_FONTSIZE
    tick_fontsize: int = DEFAULT_TICK_FONTSIZE
    legend_fontsize: int = DEFAULT_LEGEND_FONTSIZE
    show_replicate_points: bool = True
    show_std: bool = True
    
    def __post_init__(self):
        # Convert list to tuple if loaded from YAML
        if isinstance(self.figsize, list):
            self.figsize = tuple(self.figsize)


@dataclass
class VisualizationConfig:
    """
    Top-level configuration for the visualization pipeline.
    
    This is the main configuration class that ties together all settings
    needed to run the visualization pipeline.
    
    Example usage:
        config = VisualizationConfig.from_yaml("configs/source_3.yaml")
        pipeline = VisualizationPipeline(config)
        pipeline.run()
    """
    source: str
    architecture: str
    augmentation: bool
    mappings: SourceMappings
    experiments: List[ExperimentPaths]
    splits: List[str] = field(default_factory=lambda: ["batch", "plate", "well", "random"])
    plot_settings: PlotSettings = field(default_factory=PlotSettings)
    output_dir: Optional[str] = None
    base_dir: Optional[str] = None  # Base directory for relative paths
    
    @classmethod
    def from_yaml(cls, path: Path) -> "VisualizationConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file
            
        Returns:
            VisualizationConfig instance
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML is malformed
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Parse nested dataclasses
        data["mappings"] = SourceMappings(**data["mappings"])
        data["experiments"] = [
            ExperimentPaths(**e) for e in data["experiments"]
        ]
        
        if "plot_settings" in data:
            data["plot_settings"] = PlotSettings(**data["plot_settings"])
        
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            path: Path to save the YAML file
        """
        data = {
            "source": self.source,
            "architecture": self.architecture,
            "augmentation": self.augmentation,
            "mappings": {
                "labels": self.mappings.labels,
                "plate_ids": self.mappings.plate_ids,
                "well_ids": self.mappings.well_ids,
                "batch_ids": self.mappings.batch_ids,
            },
            "experiments": [
                {"split_name": e.split_name, "npy_paths": e.npy_paths}
                for e in self.experiments
            ],
            "splits": self.splits,
            "plot_settings": {
                "figsize": list(self.plot_settings.figsize),
                "dpi": self.plot_settings.dpi,
                "label_fontsize": self.plot_settings.label_fontsize,
                "tick_fontsize": self.plot_settings.tick_fontsize,
                "legend_fontsize": self.plot_settings.legend_fontsize,
                "show_replicate_points": self.plot_settings.show_replicate_points,
                "show_std": self.plot_settings.show_std,
            },
            "output_dir": self.output_dir,
            "base_dir": self.base_dir,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get_output_dir(self) -> Path:
        """
        Get the output directory path.
        
        Returns:
            Path to output directory (creates default if not specified)
        """
        if self.output_dir:
            return Path(self.output_dir)
        return Path("visualization") / self.source / self.architecture
    
    def get_split_experiments(self, split: str) -> Optional[ExperimentPaths]:
        """
        Get experiment paths for a specific split.
        
        Args:
            split: Split name (e.g., "batch", "plate")
            
        Returns:
            ExperimentPaths for the split, or None if not found
        """
        for exp in self.experiments:
            if exp.split_name == split:
                return exp
        return None
