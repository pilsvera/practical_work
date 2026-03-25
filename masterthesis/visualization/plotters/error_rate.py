"""
Error rate visualization.

This module provides the ErrorRatePlotter class for creating
horizontal bar charts of error rates.
"""

from typing import List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .base import BasePlotter
from ..calculators.errors import ErrorStatistics
from ..constants import get_label_colormap


class ErrorRatePlotter(BasePlotter):
    """
    Plots error rates as horizontal bar charts.
    
    This plotter is separated from calculation logic - it takes
    pre-computed ErrorStatistics and creates visualizations.
    
    Example:
        stats = ErrorCalculator.aggregate_errors(dfs, "labels")
        plotter = ErrorRatePlotter()
        fig = plotter.plot(stats, column="labels", split="batch")
        # fig can now be saved, displayed, etc.
    """
    
    def plot(
        self,
        stats: ErrorStatistics,
        column: str,
        split: str,
        well_label_map: Optional[Dict[str, str]] = None,
        show_replicate_points: bool = None,
        show_std: bool = None,
    ) -> plt.Figure:
        """
        Create error rate bar chart.
        
        Args:
            stats: Pre-computed error statistics from ErrorCalculator
            column: Column name for y-axis label (e.g., "labels", "plate_ids")
            split: Split name for title (e.g., "batch", "plate")
            well_label_map: Optional mapping of well_ids to labels (for coloring)
            show_replicate_points: Show individual replicate means as scatter.
                                   Uses settings default if None.
            show_std: Show error bars. Uses settings default if None.
        
        Returns:
            matplotlib Figure (not saved - caller handles saving)
        """
        # Use settings defaults if not specified
        if show_replicate_points is None:
            show_replicate_points = self.settings.show_replicate_points
        if show_std is None:
            show_std = self.settings.show_std
        
        fig, ax = self.create_figure(figsize=(12, 12))
        y_pos = np.arange(len(stats.classes))
        
        # Determine bar colors
        bar_colors = self._get_bar_colors(
            stats.classes, column, well_label_map, ax
        )
        
        # Plot bars
        xerr = stats.stds if show_std else None
        ax.barh(
            y_pos, 
            stats.means, 
            xerr=xerr, 
            color=bar_colors, 
            capsize=5, 
            label='Mean Error Rate'
        )
        
        # Plot replicate points
        if show_replicate_points:
            self._add_replicate_points(ax, stats, y_pos)
        
        # Labels and formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stats.classes)
        ax.set_xlim(0, 1.2)
        ax.set_xlabel("Error Rate")
        ax.set_ylabel(column)
        self.finalize_figure(fig)
        return fig
    
    def _get_bar_colors(
        self,
        classes: List[str],
        column: str,
        well_label_map: Optional[Dict[str, str]],
        ax: plt.Axes
    ):
        """
        Determine bar colors based on column type.
        
        For well_ids, colors bars by the compound label.
        For other columns, uses a single color.
        """
        if column == "well_ids" and well_label_map:
            labels_for_wells = [
                well_label_map.get(w, "Unknown") for w in classes
            ]
            unique_labels = sorted(set(labels_for_wells))
            colormap = get_label_colormap(unique_labels)
            bar_colors = [colormap[lbl] for lbl in labels_for_wells]

            # Add legend for label colors
            self._add_label_legend(ax, colormap, unique_labels)

            return bar_colors
        elif column == "labels":
            colormap = get_label_colormap(classes)
            return [colormap[cls] for cls in classes]
        else:
            return "#E69F00"  # Orange from colorblind palette
    
    def _add_replicate_points(
        self, 
        ax: plt.Axes, 
        stats: ErrorStatistics, 
        y_pos: np.ndarray
    ):
        """Add scatter points for each replicate's mean."""
        for i, cls in enumerate(stats.classes):
            replicate_vals = stats.replicate_means.get(cls, [])
            if replicate_vals:
                ax.scatter(
                    replicate_vals, 
                    np.full(len(replicate_vals), y_pos[i]),
                    color="black", 
                    zorder=10, 
                    s=30,
                    label="Replicates" if i == 0 else None
                )
    
    def _add_label_legend(
        self, 
        ax: plt.Axes, 
        colormap: dict, 
        labels: list,
        max_labels: int = 20
    ):
        """Add color legend for labels."""
        handles = [
            Patch(color=colormap[lbl], label=str(lbl)) 
            for lbl in labels[:max_labels]
        ]
        ax.legend(
            handles=handles, 
            title="Perturbation", 
            bbox_to_anchor=(1.05, 1), 
            loc='upper left'
        )
    
    def plot_multi_split(
        self,
        split_stats: Dict[str, ErrorStatistics],
        column: str,
        architecture: str,
    ) -> plt.Figure:
        """
        Create grouped bar chart comparing error rates across splits.
        
        Args:
            split_stats: Dict mapping split name to ErrorStatistics
            column: Column name that was used for grouping
            architecture: Model architecture name for title
            
        Returns:
            matplotlib Figure
        """
        # Get all unique classes across splits
        all_classes = sorted(set().union(*[
            set(stats.classes) for stats in split_stats.values()
        ]))
        
        fig, ax = self.create_figure(figsize=(10, 8))
        y_pos = np.arange(len(all_classes))
        bar_height = 0.15
        
        for i, (split, stats) in enumerate(split_stats.items()):
            # Build values for this split (NaN for missing classes)
            values = []
            for cls in all_classes:
                if cls in stats.classes:
                    idx = stats.classes.index(cls)
                    values.append(stats.means[idx])
                else:
                    values.append(np.nan)
            
            offset = (i - len(split_stats) / 2) * bar_height + bar_height / 2
            color = self.get_split_color(split)
            ax.barh(
                y_pos + offset, 
                values, 
                height=bar_height, 
                label=split, 
                color=color
            )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_classes)
        ax.set_xlabel("Error Rate")
        ax.set_ylabel(column)
        ax.legend()
        
        self.finalize_figure(fig)
        return fig
