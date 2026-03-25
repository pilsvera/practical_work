"""
Base plotter class with shared functionality.

This module provides the base class for all plotters, including
common utilities for figure creation and styling.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.figure

from ..config import PlotSettings
from ..constants import SPLIT_COLORS, LABEL_COLORS


class BasePlotter(ABC):
    """
    Base class for all plotters.
    
    Key principle: Plotters CREATE and RETURN figures, but never SAVE them.
    Saving is handled by FileSaver at the pipeline level.
    
    This separation of concerns makes plotters:
    - Easier to test (no file I/O side effects)
    - More flexible (can display, save, or embed figures)
    - Reusable across different output contexts
    
    Example subclass:
        class MyPlotter(BasePlotter):
            def plot(self, data) -> plt.Figure:
                fig, ax = self.create_figure()
                ax.plot(data)
                return fig
    """
    
    def __init__(self, settings: PlotSettings = None):
        """
        Initialize the plotter with optional settings.
        
        Args:
            settings: Plot settings (uses defaults if not provided)
        """
        self.settings = settings or PlotSettings()
        self._apply_font_settings()
    
    def _apply_font_settings(self):
        """Apply font settings from config to matplotlib rcParams."""
        plt.rcParams.update({
            'font.size': self.settings.label_fontsize,
            'xtick.labelsize': self.settings.tick_fontsize,
            'ytick.labelsize': self.settings.tick_fontsize,
            'legend.fontsize': self.settings.legend_fontsize,
            'axes.titlesize': self.settings.label_fontsize - 4,
            'axes.labelsize': self.settings.label_fontsize,
            'xtick.major.pad': 5
        })
    
    def create_figure(
        self, 
        figsize: Tuple[int, int] = None,
        nrows: int = 1,
        ncols: int = 1,
        **subplot_kw
    ) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, list]]:
        """
        Create a new figure with axes.
        
        Args:
            figsize: Figure size (width, height). Uses settings default if not provided.
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            **subplot_kw: Additional arguments passed to plt.subplots
            
        Returns:
            Tuple of (Figure, Axes) or (Figure, list of Axes)
        """
        figsize = figsize or self.settings.figsize
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **subplot_kw)
        return fig, axes

    @staticmethod
    def finalize_figure(fig: plt.Figure):
        """Reduce y-tick font size when 8+ items, then tight_layout."""
        for ax in fig.get_axes():
            yticks = ax.get_yticklabels()
            if len(yticks) > 10:
                for label in yticks:
                    label.set_fontsize(12)
        fig.tight_layout()
    
    @property
    def split_colors(self) -> dict:
        """Get the color palette for splits."""
        return SPLIT_COLORS
    
    @property
    def label_colors(self) -> list:
        """Get the color palette for labels."""
        return LABEL_COLORS
    
    def get_split_color(self, split: str) -> str:
        """
        Get color for a specific split.
        
        Args:
            split: Split name (e.g., "batch", "plate")
            
        Returns:
            Hex color string
        """
        return self.split_colors.get(split, "#999999")
    
    @abstractmethod
    def plot(self, *args, **kwargs) -> plt.Figure:
        """
        Create and return a figure.
        
        Subclasses must implement this method.
        
        Returns:
            matplotlib Figure object
        """
        pass
