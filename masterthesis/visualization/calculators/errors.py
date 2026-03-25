"""
Error rate calculation utilities.

This module provides pure functions for calculating error rates and
aggregating them across multiple DataFrames (replicates).
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd


@dataclass
class ErrorStatistics:
    """
    Container for error rate statistics.
    
    Attributes:
        classes: List of class names/identifiers
        means: Array of mean error rates per class
        stds: Array of standard deviations per class
        replicate_means: Dict mapping class to list of per-run mean errors
    """
    classes: List[str]
    means: np.ndarray
    stds: np.ndarray
    replicate_means: Dict[str, List[float]]
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to a DataFrame for easy saving/display.
        
        Returns:
            DataFrame with columns: class, mean_error, std_error
        """
        return pd.DataFrame({
            "class": self.classes,
            "mean_error": self.means,
            "std_error": self.stds,
        })
    
    def get_replicate_dataframe(self, column_name: str = "class") -> pd.DataFrame:
        """
        Get DataFrame with per-replicate mean errors.
        
        Args:
            column_name: Name to use for the class column
            
        Returns:
            DataFrame with columns: [column_name], run, replicate_mean_error
        """
        rows = []
        for cls in self.classes:
            for run_id, mean_val in enumerate(self.replicate_means.get(cls, [])):
                rows.append({
                    column_name: cls,
                    "run": run_id,
                    "replicate_mean_error": mean_val,
                })
        return pd.DataFrame(rows)


class ErrorCalculator:
    """
    Pure calculation of error rates. No plotting, no saving.
    
    This class provides static methods for computing error statistics
    from evaluation DataFrames.
    
    Example:
        dfs = [df1, df2, df3]  # Multiple replicates
        stats = ErrorCalculator.aggregate_errors(dfs, "labels")
        print(f"Mean error: {stats.means.mean():.2%}")
    """
    
    @staticmethod
    def calculate_classwise_errors(df: pd.DataFrame, column: str) -> pd.Series:
        """
        Group errors by a column and return lists of error values.
        
        Args:
            df: DataFrame with 'errors' column
            column: Column to group by (e.g., 'plate_ids', 'labels')
        
        Returns:
            Series with index=class, values=list of error values
            
        Example:
            errors_by_plate = ErrorCalculator.calculate_classwise_errors(df, "plate_ids")
            # errors_by_plate["plate_1"] -> [0, 1, 0, 0, 1, ...]
        """
        return df.groupby(column)["errors"].apply(list)
    
    @staticmethod
    def aggregate_errors(
        dfs: List[pd.DataFrame], 
        column: str
    ) -> ErrorStatistics:
        """
        Aggregate error rates across multiple DataFrames (replicates).
        
        This is the main entry point for error rate calculations. It
        combines errors from multiple runs and computes summary statistics.
        
        Args:
            dfs: List of DataFrames from different runs/replicates
            column: Column to group by (e.g., 'labels', 'plate_ids')
        
        Returns:
            ErrorStatistics with means, stds, and per-replicate data
            
        Example:
            # Aggregate errors across 3 training runs
            stats = ErrorCalculator.aggregate_errors(
                [run1_df, run2_df, run3_df], 
                "labels"
            )
            print(stats.to_dataframe())
        """
        # Calculate per-run errors
        all_errors = [
            ErrorCalculator.calculate_classwise_errors(df, column) 
            for df in dfs
        ]
        
        # Get all unique classes across all runs
        all_classes = sorted(set().union(*[err.index for err in all_errors]))
        
        # Aggregate errors per class
        aggregated = {cls: [] for cls in all_classes}
        replicate_means = {cls: [] for cls in all_classes}
        
        for err in all_errors:
            for cls in all_classes:
                if cls in err.index:
                    error_list = err[cls]
                    aggregated[cls].extend(error_list)
                    replicate_means[cls].append(float(np.mean(error_list)))
        
        # Compute summary statistics across replicates (not individual samples)
        means = np.array([np.mean(replicate_means[cls]) for cls in all_classes])
        stds = np.array([np.std(replicate_means[cls]) for cls in all_classes])
        
        return ErrorStatistics(
            classes=all_classes,
            means=means,
            stds=stds,
            replicate_means=replicate_means
        )
    
    @staticmethod
    def calculate_overall_error(df: pd.DataFrame) -> float:
        """
        Calculate overall error rate for a single DataFrame.
        
        Args:
            df: DataFrame with 'errors' column
            
        Returns:
            Overall error rate as a float between 0 and 1
        """
        return float(np.mean(df["errors"]))
    
    @staticmethod
    def calculate_overall_accuracy(df: pd.DataFrame) -> float:
        """
        Calculate overall accuracy for a single DataFrame.
        
        Args:
            df: DataFrame with 'errors' column
            
        Returns:
            Overall accuracy as a float between 0 and 1
        """
        return 1.0 - ErrorCalculator.calculate_overall_error(df)
