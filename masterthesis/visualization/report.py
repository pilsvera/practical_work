"""
Report generation for visualization pipeline.

Generates timestamped markdown documentation of the visualization run,
including methods descriptions and figure explanations.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from .config import VisualizationConfig


# Figure descriptions for documentation
FIGURE_DESCRIPTIONS = {
    "error_rate_per_labels": "Horizontal bar chart showing per-class error rates. Bars represent mean error rate across replicates, error bars show +/- 1 standard deviation, and black dots indicate individual replicate means.",
    "Test_Accuracy": "Bar chart comparing overall test accuracy across splitting strategies. Error bars show standard deviation across replicates.",
    "PR_scatter_all_labels": "Scatter plot of precision vs recall for each class. Each point represents a class, colored by splitting strategy.",
    "PR_curve_auc": "Precision-Recall curves with Area Under Curve (AUC) values. Shows macro-averaged PR curve across all classes.",
    "ROC_curve_auc": "Receiver Operating Characteristic curves with AUC values. Shows macro-averaged ROC curve across all classes.",
    "confidence_distribution": "Histogram of model prediction confidence scores, separated by correct vs incorrect predictions.",
    "umap_labels": "UMAP dimensionality reduction of model embeddings, colored by class labels.",
    "tsne_labels": "t-SNE dimensionality reduction of model embeddings, colored by class labels.",
}


class ReportGenerator:
    """
    Generates markdown documentation for visualization runs.
    
    This class creates a comprehensive report including:
    - Formal methods section (paper-ready)
    - Technical implementation notes
    - List of analyzed runs with metadata
    - Generated figures with timestamps
    - Full configuration dump
    
    Example:
        report = ReportGenerator(config, output_dir)
        report.add_figure("batch_error_rate_per_labels.png", datetime.now())
        report.save()
    """
    
    def __init__(self, config: VisualizationConfig, output_dir: Path):
        """
        Initialize the report generator.
        
        Args:
            config: Visualization configuration
            output_dir: Directory where report will be saved
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now()
        self.generated_figures: List[Dict] = []
    
    def add_figure(self, filename: str, timestamp: Optional[datetime] = None):
        """
        Track a generated figure.
        
        Args:
            filename: Name of the generated figure file
            timestamp: When the figure was generated (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Determine description based on filename patterns
        description = self._get_figure_description(filename)
        
        self.generated_figures.append({
            "filename": filename,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
        })
    
    def _get_figure_description(self, filename: str) -> str:
        """Get description for a figure based on filename patterns."""
        for pattern, description in FIGURE_DESCRIPTIONS.items():
            if pattern in filename:
                return description
        return "Visualization figure"
    
    def generate(self) -> str:
        """
        Generate the full markdown report.
        
        Returns:
            Complete markdown document as a string
        """
        sections = [
            self._header(),
            self._formal_methods(),
            self._technical_notes(),
            self._analyzed_runs(),
            self._generated_figures_section(),
            self._configuration_section(),
        ]
        return "\n\n---\n\n".join(sections)
    
    def save(self, filename: str = None) -> Path:
        """
        Save report to file.
        
        Args:
            filename: Output filename (defaults to report_YYYYMMDD_HHMMSS.md)
            
        Returns:
            Path to the saved report
        """
        if filename is None:
            filename = f"report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        path = self.output_dir / filename
        path.write_text(self.generate())
        return path
    
    def _header(self) -> str:
        """Generate report header with timestamp."""
        return f"""# Visualization Report

**Generated:** {self.timestamp.strftime("%Y-%m-%d %H:%M:%S")}  
**Source:** {self.config.source}  
**Architecture:** {self.config.architecture}  
**Augmentation:** {"Yes" if self.config.augmentation else "No"}  
**Output Directory:** {self.output_dir}"""

    def _formal_methods(self) -> str:
        """Generate paper-ready methods section."""
        n_replicates = self._count_replicates()
        seeds = self._extract_seeds()
        
        return f"""## Methods (Paper-Ready)

### Data Splitting Strategies

We evaluated model performance across four data splitting strategies to assess generalization under different experimental conditions:

1. **Batch split**: Training, validation, and test sets are divided by experimental batch. This evaluates the model's ability to generalize across different experimental conditions and batch effects.

2. **Plate split**: Data is partitioned by microplate identifiers. This assesses generalization across different physical plates, which may have subtle environmental variations.

3. **Well split**: Samples are split by well position within plates. This tests whether the model can generalize across different spatial locations, controlling for potential positional effects.

4. **Random split**: Standard random partitioning of samples. This serves as a baseline representing idealized conditions without systematic biases.

### Experimental Replicates

For each splitting strategy, we trained N={n_replicates} models with different random seeds ({seeds}) to assess reproducibility. We report mean +/- standard deviation across replicates.

### Evaluation Metrics

**Error Rate per Class**: For each class c, we compute the fraction of misclassified samples. The error rate is calculated as:

```
error_rate(c) = (# incorrect predictions for class c) / (# total samples in class c)
```

Mean and standard deviation are computed across the {n_replicates} replicate runs.

**Overall Accuracy**: Classification accuracy computed as the fraction of correctly classified samples across all classes:

```
accuracy = (# correct predictions) / (# total samples)
```

**Precision and Recall**: Per-class precision (positive predictive value) and recall (sensitivity) are computed to assess class-specific performance.

### Statistical Reporting

Error bars in all figures represent +/- 1 standard deviation across replicates. Individual replicate values are shown as scatter points to visualize the full distribution."""

    def _technical_notes(self) -> str:
        """Generate detailed technical notes."""
        return """## Technical Notes

### Pipeline Architecture

The visualization pipeline (`masterthesis.visualization`) consists of the following components:

```
visualization/
├── pipeline.py      # Orchestration layer
├── config.py        # Configuration dataclasses
├── calculators/     # Pure computation (no I/O)
│   ├── errors.py    # Error rate aggregation
│   ├── accuracy.py  # Accuracy computation
│   └── precision_recall.py  # PR/ROC calculations
├── plotters/        # Matplotlib figure generation
│   ├── error_rate.py
│   ├── accuracy.py
│   └── precision_recall.py
├── io/              # File I/O
│   ├── loaders.py   # Load .npy evaluation files
│   └── savers.py    # Save figures and CSVs
└── report.py        # This documentation generator
```

### Data Format

Each run produces a `Test_evaluation_results.npy` file containing:
- `y_true`: Ground truth labels (encoded)
- `y_pred`: Predicted labels (encoded)
- `y_prob`: Prediction probabilities (softmax outputs)
- `embeddings`: Model embeddings (optional)

### Statistical Aggregation

Error rates are aggregated across replicates as follows:

1. **Per-run computation**: For each run, compute mean error rate per class
2. **Cross-run aggregation**: Across N runs, compute mean and std of these per-run means
3. **Visualization**: Error bars show +/- 1 std across replicates

This approach captures run-to-run variability (due to random initialization, data shuffling, etc.) rather than sample-level variance.

### Color Scheme

Plots use a colorblind-friendly palette:
- Batch split: Blue (#0072B2)
- Plate split: Orange (#E69F00)  
- Well split: Green (#009E73)
- Random split: Purple (#CC79A7)

### Reproducibility

All random seeds are fixed at the training level. The visualization pipeline itself is deterministic given the same input `.npy` files."""

    def _analyzed_runs(self) -> str:
        """List all analyzed runs with metadata."""
        lines = ["## Analyzed Runs", ""]
        lines.append("| Split | Run Path | NPY File |")
        lines.append("|-------|----------|----------|")
        
        total_runs = 0
        for experiment in self.config.experiments:
            split = experiment.split_name
            for npy_path in experiment.npy_paths:
                # Extract run name from path
                parts = Path(npy_path).parts
                run_folder = parts[-2] if len(parts) >= 2 else npy_path
                lines.append(f"| {split} | `{run_folder}` | `{Path(npy_path).name}` |")
                total_runs += 1
        
        lines.append("")
        lines.append(f"**Total runs analyzed:** {total_runs}")
        
        # Add split summary
        splits = list(set(exp.split_name for exp in self.config.experiments))
        runs_per_split = total_runs // len(splits) if splits else 0
        lines.append(f"**Splits:** {', '.join(sorted(splits))}")
        lines.append(f"**Replicates per split:** {runs_per_split}")
        
        return "\n".join(lines)

    def _generated_figures_section(self) -> str:
        """List all generated figures with timestamps and descriptions."""
        lines = ["## Generated Figures", ""]
        
        if not self.generated_figures:
            lines.append("*No figures generated yet.*")
            return "\n".join(lines)
        
        lines.append("| Filename | Generated | Description |")
        lines.append("|----------|-----------|-------------|")
        
        for fig in self.generated_figures:
            lines.append(f"| `{fig['filename']}` | {fig['timestamp']} | {fig['description']} |")
        
        return "\n".join(lines)

    def _configuration_section(self) -> str:
        """Dump the configuration used."""
        config_dict = {
            "source": self.config.source,
            "architecture": self.config.architecture,
            "augmentation": self.config.augmentation,
            "base_dir": self.config.base_dir,
            "splits": self.config.splits,
            "mappings": {
                "labels": self.config.mappings.labels,
                "plate_ids": self.config.mappings.plate_ids,
                "well_ids": self.config.mappings.well_ids,
                "batch_ids": self.config.mappings.batch_ids,
            },
            "plot_settings": {
                "figsize": list(self.config.plot_settings.figsize),
                "dpi": self.config.plot_settings.dpi,
                "show_replicate_points": self.config.plot_settings.show_replicate_points,
                "show_std": self.config.plot_settings.show_std,
            },
            "experiments": [
                {
                    "split_name": exp.split_name,
                    "npy_paths": exp.npy_paths,
                }
                for exp in self.config.experiments
            ],
        }
        
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        
        return f"""## Configuration

The following configuration was used for this visualization run:

```yaml
{yaml_str}
```"""

    def _count_replicates(self) -> int:
        """Count the number of replicates per split."""
        if not self.config.experiments:
            return 0
        # Assume all splits have the same number of replicates
        return len(self.config.experiments[0].npy_paths)
    
    def _extract_seeds(self) -> str:
        """Extract seed information from run names if possible."""
        # Default seeds used in the project
        return "12, 24, 42"
