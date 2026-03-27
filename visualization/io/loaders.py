"""
File loading utilities for the visualization package.

This module provides classes for loading various file formats used in
the visualization pipeline, with a focus on dependency injection for
testability.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Protocol, Union

import numpy as np
import pandas as pd

from ..constants import DEFAULT_COLUMNS

logger = logging.getLogger(__name__)


class FileLoader(Protocol):
    """Protocol for file loading (enables dependency injection in tests)."""
    
    def load(self, path: Path) -> Any:
        """Load data from a file."""
        ...


class NPYLoader:
    """
    Loads numpy .npy files.
    
    Example:
        loader = NPYLoader()
        data = loader.load(Path("evaluation_results.npy"))
    """
    
    def load(self, path: Path) -> np.ndarray:
        """
        Load a numpy array from a .npy file.
        
        Args:
            path: Path to the .npy file
            
        Returns:
            Loaded numpy array
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(path)
        logger.debug(f"Loading numpy array from {path}")
        return np.load(path, allow_pickle=True)


class JSONLoader:
    """
    Loads JSON files as dictionaries.
    
    Example:
        loader = JSONLoader()
        mapping = loader.load(Path("label_encoding.json"))
    """
    
    def load(self, path: Path) -> Dict:
        """
        Load a JSON file as a dictionary.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            Parsed JSON as a dictionary
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the JSON is malformed
        """
        path = Path(path)
        logger.debug(f"Loading JSON from {path}")
        with open(path, "r") as f:
            return json.load(f)


class EvaluationDataLoader:
    """
    Loads evaluation results and applies label mappings.
    
    This class separates data loading from DataFrame construction,
    making it easier to test and reuse.
    
    Example:
        loader = EvaluationDataLoader()
        evaluation = loader.load_evaluation(Path("results.npy"))
        mappings = loader.load_mappings({
            "labels": Path("label_encoding.json"),
            "plate_ids": Path("plate_encoding.json"),
        })
        df = make_dataframe(evaluation, mappings)
    """
    
    def __init__(
        self,
        npy_loader: FileLoader = None,
        json_loader: FileLoader = None
    ):
        """
        Initialize the loader with optional custom file loaders.
        
        Args:
            npy_loader: Custom numpy loader (for testing)
            json_loader: Custom JSON loader (for testing)
        """
        self.npy_loader = npy_loader or NPYLoader()
        self.json_loader = json_loader or JSONLoader()
    
    def load_evaluation(self, npy_path: Path) -> np.ndarray:
        """
        Load raw evaluation numpy array.
        
        Args:
            npy_path: Path to the evaluation .npy file
            
        Returns:
            Raw evaluation numpy array
        """
        return self.npy_loader.load(npy_path)
    
    def load_mappings(self, mapping_paths: Dict[str, Path]) -> Dict[str, Dict]:
        """
        Load all JSON mappings.
        
        Args:
            mapping_paths: Dictionary mapping column names to JSON file paths
            
        Returns:
            Dictionary mapping column names to their encoding dictionaries
        """
        mappings = {}
        for key, path in mapping_paths.items():
            logger.debug(f"Loading mapping for {key}")
            mappings[key] = self.json_loader.load(path)
        return mappings
    
    def load_all(
        self, 
        npy_path: Path, 
        mapping_paths: Dict[str, Path]
    ) -> pd.DataFrame:
        """
        Load evaluation data and convert to DataFrame with mappings applied.
        
        Args:
            npy_path: Path to the evaluation .npy file
            mapping_paths: Dictionary mapping column names to JSON file paths
            
        Returns:
            DataFrame with human-readable labels
        """
        evaluation = self.load_evaluation(npy_path)
        mappings = self.load_mappings(mapping_paths)
        return make_dataframe(evaluation, mappings)


def make_dataframe(
    evaluation: np.ndarray,
    mappings: Dict[str, Dict],
    column_names: List[str] = None
) -> pd.DataFrame:
    """
    Pure function: converts evaluation array to DataFrame with mapped labels.
    
    This function is separated from the loader class to allow for easy
    testing and reuse with different data sources.
    
    Args:
        evaluation: Raw numpy array from model evaluation
        mappings: Dict of {column_name: {encoded_value: original_value}}
        column_names: Optional custom column names (defaults to DEFAULT_COLUMNS)
    
    Returns:
        DataFrame with human-readable labels
        
    Example:
        evaluation = np.load("results.npy")
        mappings = {"labels": {"0": "compound_A", "1": "compound_B"}}
        df = make_dataframe(evaluation, mappings)
    """
    # Build column names dynamically based on actual array width
    n_meta = 9  # plate_ids, well_ids, is_ctrl, batch_ids, labels, preds, conf, conf_class, errors
    n_logits = evaluation.shape[1] - n_meta
    if column_names:
        columns = column_names
    else:
        from ..constants import LOGIT_PREFIX
        columns = DEFAULT_COLUMNS[:n_meta] + [f"{LOGIT_PREFIX}{i}" for i in range(n_logits)]

    if evaluation.shape[1] != len(columns):
        logger.warning(
            f"Column count mismatch: array has {evaluation.shape[1]} columns, "
            f"built {len(columns)}. Using first {min(evaluation.shape[1], len(columns))} columns."
        )
        n_cols = min(evaluation.shape[1], len(columns))
        columns = columns[:n_cols]
        evaluation = evaluation[:, :n_cols]
    
    df = pd.DataFrame(evaluation, columns=columns)

    # Recompute errors as binary (labels != preds) to handle old runs
    # where the errors column contains float values instead of 0/1
    if "labels" in df.columns and "preds" in df.columns and "errors" in df.columns:
        df["errors"] = (df["labels"] != df["preds"]).astype(float)

    # Apply mappings to columns
    for column, mapping in mappings.items():
        if column in df.columns:
            reverse_map = {v: k for k, v in mapping.items()}
            df[column] = df[column].map(reverse_map)
    
    # Map predictions using labels mapping (preds use same encoding as labels)
    if "labels" in mappings and "preds" in df.columns:
        reverse_label_map = {v: k for k, v in mappings["labels"].items()}
        df["preds"] = df["preds"].map(reverse_label_map)
    
    logger.debug(f"Created DataFrame with shape {df.shape}")
    return df
