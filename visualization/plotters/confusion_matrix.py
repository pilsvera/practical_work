"""
Confusion matrix visualization.

This module provides the ConfusionMatrixPlotter class for creating
confusion matrix heatmaps from evaluation results.
"""

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .base import BasePlotter


class ConfusionMatrixPlotter(BasePlotter):
    """
    Plots confusion matrices as annotated heatmaps.

    Example:
        plotter = ConfusionMatrixPlotter()
        fig = plotter.plot_confusion_matrix(df, title="Batch split")
    """

    def plot(self, df: pd.DataFrame, **kwargs) -> plt.Figure:
        """Shorthand entry point — delegates to plot_confusion_matrix."""
        return self.plot_confusion_matrix(df, **kwargs)

    def plot_confusion_matrix(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        architecture: Optional[str] = None,
        split: Optional[str] = None,
        normalize: bool = True,
    ) -> plt.Figure:
        """
        Create a confusion matrix heatmap.

        Args:
            df: DataFrame with 'labels' and 'preds' columns
            title: Plot title (auto-generated if None)
            architecture: Model architecture name (for auto-title)
            split: Split name (for auto-title)
            normalize: If True, normalize rows to percentages

        Returns:
            matplotlib Figure
        """
        labels_sorted = sorted(df["labels"].unique())
        cm = confusion_matrix(df["labels"], df["preds"], labels=labels_sorted)

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = cm.astype(float) / np.maximum(row_sums, 1)
            # Annotations: show percentage and count
            annot = np.empty_like(cm, dtype=object)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    annot[i, j] = f"{cm_norm[i, j]:.0%}\n({cm[i, j]})"
            plot_data = cm_norm
        else:
            annot = cm.astype(str)
            plot_data = cm

        fig, ax = self.create_figure(figsize=(10, 8))

        sns.heatmap(
            plot_data,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=labels_sorted,
            yticklabels=labels_sorted,
            ax=ax,
            vmin=0,
            vmax=1 if normalize else None,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"fontsize": 10},
        )

        if title is None:
            if architecture and split:
                title = f"{architecture} - {split} - Confusion Matrix"
            else:
                title = "Confusion Matrix"

        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        self.finalize_figure(fig)
        return fig
