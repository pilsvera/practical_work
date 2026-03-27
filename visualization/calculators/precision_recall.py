"""
Precision-recall and ROC curve calculation utilities.

This module provides pure functions for calculating precision, recall,
and related metrics from evaluation DataFrames.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_curve,
    auc,
)


@dataclass
class PRStatistics:
    """
    Precision-recall statistics per class.
    
    Attributes:
        labels: List of class labels
        precision: Array of precision values per class
        recall: Array of recall values per class
    """
    labels: List[str]
    precision: np.ndarray
    recall: np.ndarray
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to a DataFrame for easy saving/display.
        
        Returns:
            DataFrame with columns: label, precision, recall
        """
        return pd.DataFrame({
            "label": self.labels,
            "precision": self.precision,
            "recall": self.recall,
        })


@dataclass
class CurveData:
    """
    Data for PR or ROC curves.
    
    Attributes:
        x: X-axis values (recall for PR curve, FPR for ROC curve)
        y: Y-axis values (precision for PR curve, TPR for ROC curve)
        auc_score: Area under the curve
    """
    x: np.ndarray
    y: np.ndarray
    auc_score: float


class PrecisionRecallCalculator:
    """
    Pure calculation of precision, recall, and related curves.
    
    This class computes classification metrics from evaluation DataFrames.
    It has no side effects - no file I/O or plotting.
    
    Example:
        calc = PrecisionRecallCalculator(df)
        pr_stats = calc.calculate_per_class()
        roc_curve = calc.calculate_roc_curve()
        print(f"ROC AUC: {roc_curve.auc_score:.3f}")
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with an evaluation DataFrame.
        
        Args:
            df: DataFrame with 'labels', 'preds', and 'confs' columns
        """
        self.df = df
    
    def calculate_per_class(self) -> PRStatistics:
        """
        Calculate precision and recall per class.
        
        Returns:
            PRStatistics with per-class precision and recall
        """
        y_true = self.df["labels"].astype(str)
        y_pred = self.df["preds"].astype(str)
        labels_sorted = sorted(y_true.unique())
        
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=labels_sorted
        )
        
        return PRStatistics(
            labels=labels_sorted,
            precision=precision,
            recall=recall
        )
    
    def calculate_pr_curve(self) -> CurveData:
        """
        Calculate precision-recall curve based on confidence scores.
        
        This creates a binary classification problem: was the prediction
        correct (confidence should be high) or wrong (confidence should be low)?
        
        Returns:
            CurveData with recall as x, precision as y, and PR-AUC
        """
        y_true = (self.df["labels"] == self.df["preds"]).astype(int)
        scores = self.df["conf"]
        
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall, precision)
        
        return CurveData(x=recall, y=precision, auc_score=pr_auc)
    
    def calculate_roc_curve(self) -> CurveData:
        """
        Calculate ROC curve based on confidence scores.
        
        This creates a binary classification problem: was the prediction
        correct (confidence should be high) or wrong (confidence should be low)?
        
        Returns:
            CurveData with FPR as x, TPR as y, and ROC-AUC
        """
        y_true = (self.df["labels"] == self.df["preds"]).astype(int)
        scores = self.df["conf"]
        
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        
        return CurveData(x=fpr, y=tpr, auc_score=roc_auc)
    
    @staticmethod
    def aggregate_across_runs(
        dfs: List[pd.DataFrame], 
        split_name: str
    ) -> pd.DataFrame:
        """
        Aggregate precision/recall across multiple runs.
        
        Args:
            dfs: List of DataFrames from different runs
            split_name: Name of the split (e.g., "batch", "plate")
            
        Returns:
            DataFrame with columns: label, precision, recall, split, run
        """
        rows = []
        for run_id, df in enumerate(dfs):
            stats = PrecisionRecallCalculator(df).calculate_per_class()
            for i, label in enumerate(stats.labels):
                rows.append({
                    "label": label,
                    "precision": stats.precision[i],
                    "recall": stats.recall[i],
                    "split": split_name,
                    "run": run_id
                })
        return pd.DataFrame(rows)
    
    @staticmethod
    def aggregate_curves_across_runs(
        dfs: List[pd.DataFrame],
        curve_type: str = "roc"
    ) -> Tuple[CurveData, List[float]]:
        """
        Calculate curves for multiple runs and return mean AUC.
        
        Args:
            dfs: List of DataFrames from different runs
            curve_type: Either "roc" or "pr"
            
        Returns:
            Tuple of (first run's CurveData for plotting, list of all AUCs)
        """
        aucs = []
        first_curve = None
        
        for i, df in enumerate(dfs):
            calc = PrecisionRecallCalculator(df)
            if curve_type == "roc":
                curve = calc.calculate_roc_curve()
            else:
                curve = calc.calculate_pr_curve()
            
            aucs.append(curve.auc_score)
            if i == 0:
                first_curve = curve
        
        return first_curve, aucs
