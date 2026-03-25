"""
Module-level constants for column names, color palettes, and shared configurations.

This module centralizes all magic strings and color definitions to ensure
consistency across the visualization package.
"""

from typing import Dict, List

# Default column names for evaluation DataFrames
DEFAULT_COLUMNS: List[str] = [
    "plate_ids",
    "well_ids",
    "is_ctrl",
    "batch_ids",
    "labels",
    "preds",
    "conf",
    "conf_class",
    "errors",
    "logits_0",
    "logits_1",
    "logits_2",
    "logits_3",
    "logits_4",
    "logits_5",
    "logits_6",
    "logits_7",
]

# Prefix for logit columns (for dynamic detection)
LOGIT_PREFIX: str = "logits_"

# Colorblind-friendly palette for splits
SPLIT_COLORS: Dict[str, str] = {
    "batch": "#efb637",
    "plate": "#8bc34a",
    "well": "#7bd0c1",
    "random": "#e9769f",
}

# Colorblind-friendly palette for labels (Wong palette)
LABEL_COLORS: List[str] = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#999999",  # Gray
]

# Fixed label-to-color mapping (Wong colorblind-friendly palette)
LABEL_COLOR_MAP: Dict[str, str] = {
    "AMG900":       "#E69F00",  # Orange
    "FK-866":       "#56B4E9",  # Sky Blue
    "LY2109761":    "#009E73",  # Bluish Green
    "NVS-PAK1-1":   "#F0E442",  # Yellow
    "TC-S-7004":    "#0072B2",  # Blue
    "aloxistatin":  "#D55E00",  # Vermillion
    "dexamethasone":"#CC79A7",  # Reddish Purple
    "quinidine":    "#999999",  # Gray
}

# Default plot settings
DEFAULT_FIGSIZE: tuple = (12, 12)
DEFAULT_DPI: int = 150
DEFAULT_LABEL_FONTSIZE: int = 22
DEFAULT_TITLE_FONTSIZE: int = 24
DEFAULT_TICK_FONTSIZE: int = 18
DEFAULT_LEGEND_FONTSIZE: int = 18


def get_label_colormap(unique_labels: List) -> Dict:
    """
    Generate a color mapping for labels using the colorblind-friendly palette.

    Uses the fixed LABEL_COLOR_MAP for known compound labels, falling back
    to cycling through LABEL_COLORS for unknown labels.

    Args:
        unique_labels: List of unique label values to map

    Returns:
        Dictionary mapping each label to a color hex string
    """
    colormap = {}
    fallback_idx = 0
    for lbl in sorted(unique_labels):
        if str(lbl) in LABEL_COLOR_MAP:
            colormap[lbl] = LABEL_COLOR_MAP[str(lbl)]
        else:
            colormap[lbl] = LABEL_COLORS[fallback_idx % len(LABEL_COLORS)]
            fallback_idx += 1
    return colormap


def get_logit_columns(n_logits: int = 8) -> List[str]:
    """
    Generate logit column names.
    
    Args:
        n_logits: Number of logit columns
        
    Returns:
        List of logit column names
    """
    return [f"{LOGIT_PREFIX}{i}" for i in range(n_logits)]
