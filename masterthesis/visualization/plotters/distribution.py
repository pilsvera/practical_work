"""
Distribution visualizations.

This module provides the DistributionPlotter class for creating
histograms and density plots of various metrics.
"""

from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BasePlotter


class DistributionPlotter(BasePlotter):
    """
    Plots distribution visualizations.
    
    Example:
        plotter = DistributionPlotter()
        fig = plotter.plot_confidence(df, title="Confidence Distribution")
    """
    
    def plot(
        self,
        df: pd.DataFrame,
        column: str = "conf",
        **kwargs
    ) -> plt.Figure:
        """
        Default plot method - creates confidence distribution.
        
        Args:
            df: DataFrame with the specified column and 'errors'
            column: Column to plot distribution of
            **kwargs: Additional arguments passed to plot_confidence
            
        Returns:
            matplotlib Figure
        """
        return self.plot_confidence(df, **kwargs)
    
    def plot_confidence(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        architecture: Optional[str] = None,
        split: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot confidence distribution for correct vs wrong predictions.
        
        Args:
            df: DataFrame with 'confs' and 'errors' columns
            title: Plot title (auto-generated if None)
            architecture: Model architecture name (for auto-title)
            split: Split name (for auto-title)
        
        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(figsize=(10, 6))
        
        # Convert errors to string labels for better legend
        df = df.copy()
        # errors may be binary (0/1) or float; derive from labels vs preds if available
        if "labels" in df.columns and "preds" in df.columns:
            df["Prediction"] = (df["labels"] == df["preds"]).map({True: "Correct", False: "Wrong"})
        else:
            df["Prediction"] = df["errors"].apply(lambda x: "Correct" if x == 0 else "Wrong")
        
        sns.histplot(
            data=df, 
            x="conf", 
            hue="Prediction",
            bins=30, 
            kde=True, 
            ax=ax,
            palette={"Correct": "#009E73", "Wrong": "#D55E00"}
        )
        
        # Generate title
        if title is None:
            if architecture and split:
                title = f"{architecture} - {split} - Confidence Distribution"
            else:
                title = "Confidence Distribution: Correct vs Wrong"
        
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Frequency")
        
        self.finalize_figure(fig)
        return fig
    
    def plot_error_distribution(
        self,
        df: pd.DataFrame,
        group_by: str = "labels",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot error rate distribution across groups.
        
        Args:
            df: DataFrame with 'errors' and group_by column
            group_by: Column to group by
            title: Plot title
        
        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(figsize=(10, 6))
        
        # Calculate error rates per group
        error_rates = df.groupby(group_by)["errors"].mean()
        
        sns.histplot(
            error_rates, 
            bins=20, 
            kde=True, 
            ax=ax,
            color="#0072B2"
        )
        
        ax.set_xlabel("Error Rate")
        ax.set_ylabel("Count")
        
        # Add mean line
        mean_error = error_rates.mean()
        ax.axvline(
            mean_error, 
            color='red', 
            linestyle='--', 
            label=f'Mean: {mean_error:.3f}'
        )
        ax.legend()
        
        self.finalize_figure(fig)
        return fig
