"""
Dimensionality reduction visualizations.

This module provides the DimensionalityPlotter class for creating
UMAP and t-SNE visualizations.
"""

from typing import Optional, List
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .base import BasePlotter
from ..constants import LOGIT_PREFIX, get_label_colormap

logger = logging.getLogger(__name__)

# Optional import for UMAP (may not be installed)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")


class DimensionalityPlotter(BasePlotter):
    """
    Plots dimensionality reduction visualizations (UMAP, t-SNE).
    
    Example:
        plotter = DimensionalityPlotter()
        fig = plotter.plot_umap(df, color_by="labels")
        fig = plotter.plot_tsne(df, title="t-SNE of Logits")
    """
    
    def plot(
        self,
        df: pd.DataFrame,
        method: str = "umap",
        color_by: str = "labels",
        **kwargs
    ) -> plt.Figure:
        """
        Default plot method - creates dimensionality reduction plot.
        
        Args:
            df: DataFrame with logit columns and metadata
            method: Either "umap" or "tsne"
            color_by: Column to use for coloring points
            **kwargs: Additional arguments passed to specific plot method
            
        Returns:
            matplotlib Figure
        """
        if method == "umap":
            return self.plot_umap(df, color_by=color_by, **kwargs)
        else:
            return self.plot_tsne(df, **kwargs)
    
    def _extract_logits(self, df: pd.DataFrame) -> np.ndarray:
        """Extract logit columns from DataFrame."""
        logit_cols = [c for c in df.columns if c.startswith(LOGIT_PREFIX)]
        if not logit_cols:
            raise ValueError(
                f"No columns starting with '{LOGIT_PREFIX}' found in DataFrame"
            )
        return df[logit_cols].values
    
    def plot_umap(
        self,
        df: pd.DataFrame,
        color_by: str = "labels",
        title: Optional[str] = None,
        logits: Optional[np.ndarray] = None,
        n_components: int = 2,
        random_state: int = 42,
    ) -> plt.Figure:
        """
        Create UMAP visualization.
        
        Args:
            df: DataFrame with logit columns and metadata
            color_by: Column to use for coloring points
            title: Plot title (auto-generated if None)
            logits: Pre-extracted logits (extracted from df if None)
            n_components: Number of UMAP components
            random_state: Random seed for reproducibility
        
        Returns:
            matplotlib Figure
        """
        if not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP is not installed. Install with: pip install umap-learn"
            )
        
        fig, ax = self.create_figure(figsize=(10, 8))
        
        logits = logits if logits is not None else self._extract_logits(df)
        
        logger.info(f"Fitting UMAP on {len(df)} samples...")
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        embedding = reducer.fit_transform(logits)
        
        unique_values = sorted(df[color_by].unique())
        colormap = get_label_colormap(unique_values)
        
        for val in unique_values:
            mask = df[color_by] == val
            ax.scatter(
                embedding[mask, 0], 
                embedding[mask, 1], 
                label=val, 
                alpha=0.3, 
                s=30,
                color=colormap.get(val)
            )
        
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="small")
        
        self.finalize_figure(fig)
        return fig
    
    def plot_umap_by_plate(
        self,
        df: pd.DataFrame,
        split_label: str,
        logits: Optional[np.ndarray] = None,
    ) -> plt.Figure:
        """
        Create UMAP visualization colored by plate ID.
        
        Args:
            df: DataFrame with logit columns and plate_ids
            split_label: Split name for title
            logits: Pre-extracted logits (extracted from df if None)
        
        Returns:
            matplotlib Figure
        """
        return self.plot_umap(
            df, 
            color_by="plate_ids", 
            title=f"UMAP by Plate ID: {split_label}",
            logits=logits
        )
    
    def plot_tsne(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        color_by_error: bool = True,
        random_state: int = 42,
    ) -> plt.Figure:
        """
        Create t-SNE visualization.

        Args:
            df: DataFrame with logit columns, labels, and errors
            title: Plot title (auto-generated if None)
            color_by_error: If True, color by error; if False, color by label
            random_state: Random seed for reproducibility

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(figsize=(10, 8))

        logits = self._extract_logits(df)

        logger.info(f"Fitting t-SNE on {len(df)} samples...")
        tsne = TSNE(n_components=2, random_state=random_state)
        proj = tsne.fit_transform(logits)

        # Use different markers for different labels
        marker_styles = ['o', 's', '^', 'v', 'P', 'X', 'D', '*', '<', '>']
        labels = df["labels"].unique()
        label_marker_map = {
            lbl: marker_styles[i % len(marker_styles)]
            for i, lbl in enumerate(labels)
        }

        scatter = None
        for label in labels:
            mask = df["labels"] == label
            scatter = ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=df.loc[mask, "errors"] if color_by_error else None,
                cmap="coolwarm" if color_by_error else None,
                alpha=0.6,
                marker=label_marker_map[label],
                label=str(label),
                edgecolors='w',
                linewidths=0.5
            )

        if color_by_error and scatter is not None:
            plt.colorbar(scatter, ax=ax, label="Error")

        ax.legend(title="Labels", bbox_to_anchor=(1.35, 1), loc='upper left', fontsize="small")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        self.finalize_figure(fig)
        logger.info("t-SNE plot complete")
        return fig
