"""
Accuracy visualization.

This module provides the AccuracyPlotter class for creating
accuracy comparison bar charts.
"""

from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base import BasePlotter
from ..constants import SPLIT_COLORS


class AccuracyPlotter(BasePlotter):
    """
    Plots accuracy comparison visualizations.
    
    Example:
        plotter = AccuracyPlotter()
        fig = plotter.plot_accuracy_bars(summary_df)
    """
    
    def plot(
        self,
        summary_df: pd.DataFrame,
        **kwargs
    ) -> plt.Figure:
        """
        Default plot method - creates accuracy bar chart.
        
        Args:
            summary_df: DataFrame with columns: split, mean_accuracy, std_accuracy
            **kwargs: Additional arguments passed to plot_accuracy_bars
            
        Returns:
            matplotlib Figure
        """
        return self.plot_accuracy_bars(summary_df, **kwargs)
    
    def plot_accuracy_bars(
        self,
        summary_df: pd.DataFrame,
        title: str = "Test Accuracy",
        order: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Plot horizontal bar chart of test accuracy per split.
        
        Args:
            summary_df: DataFrame with columns: split, mean_accuracy, std_accuracy
            title: Plot title
            order: Optional list specifying split order
        
        Returns:
            matplotlib Figure
        """
        # Apply ordering if specified
        if order:
            order = [s for s in order if s in summary_df["split"].values]
            summary_df = summary_df.copy()
            summary_df["split"] = pd.Categorical(
                summary_df["split"], 
                categories=order, 
                ordered=True
            )
            summary_df = summary_df.sort_values("split").reset_index(drop=True)
        
        fig, ax = self.create_figure(figsize=(9, 3.8))
        
        y = np.arange(len(summary_df))
        means = summary_df["mean_accuracy"].values
        stds = summary_df["std_accuracy"].fillna(0).values
        
        # Get colors for each split
        bar_colors = [
            SPLIT_COLORS.get(s, "#999999") 
            for s in summary_df["split"].astype(str).tolist()
        ]
        
        ax.barh(
            y, 
            means, 
            xerr=stds, 
            capsize=12, 
            alpha=0.95, 
            edgecolor="none", 
            color=bar_colors
        )
        
        # Labels with "splits: " prefix
        ytick_labels = [f"splits: {s}" for s in summary_df["split"].astype(str)]
        ax.set_yticks(y)
        ax.set_yticklabels(ytick_labels)
        ax.set_xlabel("Accuracy")
        
        # Set x-axis limit
        max_val = float(np.nanmax(means + stds))
        ax.set_xlim(0.0, max(0.65, max_val + 0.05))
        
        self.finalize_figure(fig)
        return fig
    
    def plot_per_run_accuracy(
        self,
        per_run_df: pd.DataFrame,
        title: str = "Per-Run Accuracy",
    ) -> plt.Figure:
        """
        Plot box/violin plot of accuracy across runs for each split.
        
        Args:
            per_run_df: DataFrame with columns: split, run, accuracy
            title: Plot title
        
        Returns:
            matplotlib Figure
        """
        import seaborn as sns
        
        fig, ax = self.create_figure(figsize=(10, 6))
        
        # Create violin plot
        palette = {
            split: SPLIT_COLORS.get(split, "#999999") 
            for split in per_run_df["split"].unique()
        }
        
        sns.violinplot(
            data=per_run_df, 
            x="split", 
            y="accuracy",
            palette=palette,
            inner="box",
            ax=ax
        )
        
        # Add individual points
        sns.stripplot(
            data=per_run_df,
            x="split",
            y="accuracy",
            color="black",
            size=8,
            alpha=0.6,
            ax=ax
        )
        
        ax.set_xlabel("Split")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        
        self.finalize_figure(fig)
        return fig
    
    def plot_accuracy_comparison(
        self,
        split_stats: Dict[str, pd.DataFrame],
        group_by: str = "labels",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot grouped bar chart comparing accuracy across splits.
        
        Args:
            split_stats: Dict mapping split name to accuracy DataFrame
            group_by: Column used for grouping
            title: Plot title
        
        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(figsize=(12, 8))
        
        # Get all unique classes
        all_classes = sorted(set().union(*[
            set(df[group_by].unique()) for df in split_stats.values()
        ]))
        
        x = np.arange(len(all_classes))
        width = 0.8 / len(split_stats)
        
        for i, (split, df) in enumerate(split_stats.items()):
            # Calculate accuracy per class
            acc_by_class = df.groupby(group_by).apply(
                lambda g: 1.0 - g["errors"].mean()
            )
            
            values = [acc_by_class.get(cls, np.nan) for cls in all_classes]
            offset = (i - len(split_stats) / 2 + 0.5) * width
            color = SPLIT_COLORS.get(split, "#999999")
            
            ax.bar(
                x + offset, 
                values, 
                width * 0.9, 
                label=split, 
                color=color
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=45, ha='right')
        ax.set_xlabel(group_by)
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.set_ylim(0, 1.05)
        
        self.finalize_figure(fig)
        return fig
