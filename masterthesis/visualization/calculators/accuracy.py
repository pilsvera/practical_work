"""
Accuracy calculation utilities.

This module provides pure functions for computing accuracy statistics
across multiple splits and runs.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_split_accuracies(
    split_to_dfs: Dict[str, List[pd.DataFrame]]
) -> pd.DataFrame:
    """
    Compute accuracy per run for each split.
    
    Args:
        split_to_dfs: Dict mapping split name to list of DataFrames
                      (one DataFrame per run/replicate)
    
    Returns:
        DataFrame with columns: split, run, accuracy
        
    Example:
        split_data = {
            "batch": [df_run1, df_run2, df_run3],
            "plate": [df_run1, df_run2, df_run3],
        }
        per_run = compute_split_accuracies(split_data)
        # Returns DataFrame with 6 rows (3 runs * 2 splits)
    """
    rows = []
    for split, df_group in split_to_dfs.items():
        for run_id, df in enumerate(df_group):
            acc = 1.0 - float(np.mean(df["errors"]))
            rows.append({
                "split": split,
                "run": run_id,
                "accuracy": acc
            })
    return pd.DataFrame(rows)


def summarize_accuracies(
    per_run_df: pd.DataFrame,
    order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Summarize per-run accuracies to mean/std per split.
    
    Args:
        per_run_df: Output from compute_split_accuracies
        order: Optional list specifying split order (for display)
    
    Returns:
        DataFrame with columns: split, mean_accuracy, std_accuracy
        
    Example:
        per_run = compute_split_accuracies(split_data)
        summary = summarize_accuracies(per_run, order=["batch", "plate", "well", "random"])
    """
    summary = (
        per_run_df
        .groupby("split", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std")
        )
    )
    
    # Apply ordering if specified
    if order:
        # Keep only splits that exist, in the specified order
        order = [s for s in order if s in summary["split"].unique()]
        summary["split"] = pd.Categorical(
            summary["split"], 
            categories=order, 
            ordered=True
        )
        summary = summary.sort_values("split").reset_index(drop=True)
    
    return summary


def compute_per_class_accuracy(
    df: pd.DataFrame,
    class_column: str = "labels"
) -> pd.DataFrame:
    """
    Compute accuracy per class.
    
    Args:
        df: DataFrame with 'errors' column
        class_column: Column to group by (default: 'labels')
        
    Returns:
        DataFrame with columns: [class_column], accuracy, n_samples
    """
    grouped = df.groupby(class_column).agg(
        total_errors=("errors", "sum"),
        n_samples=("errors", "count")
    ).reset_index()
    
    grouped["accuracy"] = 1.0 - (grouped["total_errors"] / grouped["n_samples"])
    
    return grouped[[class_column, "accuracy", "n_samples"]]
