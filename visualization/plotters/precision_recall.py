"""
Precision-recall and ROC curve visualizations.

This module provides the PrecisionRecallPlotter class for creating
precision-recall scatter plots and curves.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base import BasePlotter
from ..calculators.precision_recall import CurveData


class PrecisionRecallPlotter(BasePlotter):
    """
    Plots precision-recall scatter plots and curves.
    
    Example:
        plotter = PrecisionRecallPlotter()
        fig = plotter.plot_scatter(pr_df, "ResNet50")
        fig = plotter.plot_pr_curves(curves_by_split, "ResNet50")
    """
    
    def plot(
        self,
        pr_df: pd.DataFrame,
        architecture: str
    ) -> plt.Figure:
        """
        Default plot method - creates scatter plot.
        
        Args:
            pr_df: DataFrame with columns: label, precision, recall, split
            architecture: Model architecture name for title
            
        Returns:
            matplotlib Figure
        """
        return self.plot_scatter(pr_df, architecture)
    
    def plot_scatter(
        self,
        pr_df: pd.DataFrame,
        architecture: str
    ) -> plt.Figure:
        """
        Plot precision vs recall scatter for all classes and splits.
        
        Args:
            pr_df: DataFrame with columns: label, precision, recall, split
            architecture: Model architecture name for title
            
        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(figsize=(10, 8))
        
        for split in pr_df["split"].unique():
            subset = pr_df[pr_df["split"] == split]
            color = self.get_split_color(split)
            ax.scatter(
                subset["precision"], 
                subset["recall"], 
                s=70, 
                alpha=0.7,
                label=split, 
                color=color
            )
        
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend(title="Split")
        
        self.finalize_figure(fig)
        return fig
    
    def plot_per_class(
        self,
        pr_df: pd.DataFrame,
        architecture: str,
        class_label: str
    ) -> plt.Figure:
        """
        Plot precision vs recall for a single class across splits.
        
        Args:
            pr_df: DataFrame with columns: label, precision, recall, split
            architecture: Model architecture name for title
            class_label: Class label to plot
            
        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(figsize=(6, 6))
        
        subset = pr_df[pr_df["label"] == class_label]
        for split in subset["split"].unique():
            split_data = subset[subset["split"] == split]
            color = self.get_split_color(split)
            ax.scatter(
                split_data["precision"], 
                split_data["recall"], 
                label=split, 
                color=color, 
                s=100, 
                alpha=0.7
            )
        
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend(title="Split")
        
        self.finalize_figure(fig)
        return fig
    
    def plot_pr_curves(
        self,
        curves: Dict[str, CurveData],
        architecture: str
    ) -> plt.Figure:
        """
        Plot PR curves for multiple splits.
        
        Args:
            curves: Dict mapping split name to CurveData
            architecture: Model architecture name for title
            
        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(figsize=(10, 8))
        
        for split, curve in curves.items():
            color = self.get_split_color(split)
            ax.plot(
                curve.x, 
                curve.y,
                label=f"{split} (AUC={curve.auc_score:.2f})",
                color=color,
                linewidth=2
            )
        
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True)
        ax.legend(title="Split")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        
        self.finalize_figure(fig)
        return fig
    
    def plot_roc_curves(
        self,
        curves: Dict[str, CurveData],
        architecture: str
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple splits.
        
        Args:
            curves: Dict mapping split name to CurveData
            architecture: Model architecture name for title
            
        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(figsize=(10, 8))
        
        for split, curve in curves.items():
            color = self.get_split_color(split)
            ax.plot(
                curve.x, 
                curve.y,
                label=f"{split} (AUC={curve.auc_score:.2f})",
                color=color,
                linewidth=2
            )
        
        # Chance line
        ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Chance')
        
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True)
        ax.legend(title="Split")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        
        self.finalize_figure(fig)
        return fig
