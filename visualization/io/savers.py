"""
File saving utilities for the visualization package.

This module provides a unified interface for saving various output formats
including figures, DataFrames, and JSON data.
"""

import json
import logging
from pathlib import Path
from typing import Any, List, Union

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure

logger = logging.getLogger(__name__)


class FileSaver:
    """
    Handles all file saving operations.
    
    This class provides a unified interface for saving various output formats
    and manages output directory creation.
    
    Example:
        saver = FileSaver(Path("output/plots"))
        saver.save_figure(fig, "error_rate.png")
        saver.save_dataframe(df, "statistics.csv")
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the saver with a base output directory.
        
        Args:
            base_dir: Base directory for all saved files
        """
        self.base_dir = Path(base_dir)
    
    def ensure_dir(self) -> Path:
        """
        Create output directory if it doesn't exist.
        
        Returns:
            Path to the output directory
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {self.base_dir}")
        return self.base_dir
    
    def get_path(self, filename: str) -> Path:
        """
        Get full path for a filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Full path including base directory
        """
        return self.base_dir / filename
    
    def save_dataframe(
        self, 
        df: pd.DataFrame, 
        filename: str,
        index: bool = False
    ) -> Path:
        """
        Save DataFrame as CSV.
        
        Args:
            df: DataFrame to save
            filename: Output filename (should end with .csv)
            index: Whether to include row index
            
        Returns:
            Path to the saved file
        """
        self.ensure_dir()
        path = self.get_path(filename)
        df.to_csv(path, index=index)
        logger.info(f"Saved DataFrame to {path}")
        return path
    
    def save_json(
        self, 
        obj: Any, 
        filename: str,
        indent: int = 2
    ) -> Path:
        """
        Save object as JSON.
        
        Args:
            obj: Object to serialize (must be JSON-serializable)
            filename: Output filename (should end with .json)
            indent: JSON indentation level
            
        Returns:
            Path to the saved file
        """
        self.ensure_dir()
        path = self.get_path(filename)
        with open(path, "w") as f:
            json.dump(obj, f, indent=indent)
        logger.info(f"Saved JSON to {path}")
        return path
    
    def save_figure(
        self, 
        fig: matplotlib.figure.Figure, 
        filename: str,
        dpi: int = 150,
        bbox_inches: str = "tight",
        close_fig: bool = True
    ) -> Path:
        """
        Save matplotlib figure.
        
        Args:
            fig: Matplotlib figure to save
            filename: Output filename (e.g., "plot.png")
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box setting ("tight" removes whitespace)
            close_fig: Whether to close the figure after saving
            
        Returns:
            Path to the saved file
        """
        self.ensure_dir()
        path = self.get_path(filename)
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Saved figure to {path}")
        
        if close_fig:
            plt.close(fig)
        
        return path
    
    def save_text(
        self, 
        lines: List[str], 
        filename: str
    ) -> Path:
        """
        Save lines as text file.
        
        Args:
            lines: List of lines to write
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        self.ensure_dir()
        path = self.get_path(filename)
        with open(path, "w") as f:
            for line in lines:
                f.write(f"{line}\n")
        logger.info(f"Saved text to {path}")
        return path
    
    def save_array_summary(
        self,
        values: List[float],
        labels: List[str],
        filename: str
    ) -> Path:
        """
        Save array values with labels as text file.
        
        Args:
            values: List of numeric values
            labels: List of labels corresponding to values
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        lines = [f"{label}: {value}" for label, value in zip(labels, values)]
        return self.save_text(lines, filename)
