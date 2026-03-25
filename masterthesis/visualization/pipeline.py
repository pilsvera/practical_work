"""
Orchestration layer for the visualization pipeline.

This module connects loaders, calculators, plotters, and savers
to provide a unified interface for generating visualizations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .config import VisualizationConfig, PlotSettings
from .io.loaders import EvaluationDataLoader, make_dataframe
from .io.savers import FileSaver
from .calculators.errors import ErrorCalculator
from .calculators.precision_recall import PrecisionRecallCalculator
from .calculators.accuracy import compute_split_accuracies, summarize_accuracies
from .plotters.error_rate import ErrorRatePlotter
from .plotters.precision_recall import PrecisionRecallPlotter
from .plotters.dimensionality import DimensionalityPlotter
from .plotters.distribution import DistributionPlotter
from .plotters.accuracy import AccuracyPlotter
from .plotters.confusion_matrix import ConfusionMatrixPlotter
from .report import ReportGenerator

logger = logging.getLogger(__name__)


# Available plot types
PLOT_TYPES = [
    "error_rate",
    "accuracy",
    "precision_recall",
    "pr_curve",
    "roc_curve",
    "umap",
    "tsne",
    "confidence",
    "confusion_matrix",
]


class VisualizationPipeline:
    """
    Orchestrates the visualization workflow.
    
    This class ties together all components of the visualization system:
    - Loads data using EvaluationDataLoader
    - Computes statistics using calculator classes
    - Generates figures using plotter classes
    - Saves outputs using FileSaver
    
    Example:
        config = VisualizationConfig.from_yaml("config.yaml")
        pipeline = VisualizationPipeline(config)
        pipeline.run(plots=["error_rate", "accuracy"])
    """
    
    def __init__(
        self, 
        config: VisualizationConfig, 
        dry_run: bool = False,
        generate_report: bool = False
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Visualization configuration
            dry_run: If True, don't save files (useful for testing)
            generate_report: If True, generate markdown documentation report
        """
        self.config = config
        self.dry_run = dry_run
        self.generate_report = generate_report
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize components
        self.loader = EvaluationDataLoader()
        self.saver = FileSaver(config.get_output_dir())
        
        # Initialize plotters with shared settings
        self.error_plotter = ErrorRatePlotter(config.plot_settings)
        self.pr_plotter = PrecisionRecallPlotter(config.plot_settings)
        self.dim_plotter = DimensionalityPlotter(config.plot_settings)
        self.dist_plotter = DistributionPlotter(config.plot_settings)
        self.acc_plotter = AccuracyPlotter(config.plot_settings)
        self.cm_plotter = ConfusionMatrixPlotter(config.plot_settings)
        
        # Report generator (optional)
        self.report: Optional[ReportGenerator] = None
        if generate_report:
            self.report = ReportGenerator(config, Path(config.get_output_dir()))
        
        # Data cache
        self._dataframes: Dict[str, List[pd.DataFrame]] = {}
        self._mappings: Optional[Dict[str, Dict]] = None
    
    def _load_mappings(self) -> Dict[str, Dict]:
        """Load label mappings from JSON files."""
        if self._mappings is not None:
            return self._mappings
        
        mapping_paths = self.config.mappings.to_path_dict()
        self._mappings = self.loader.load_mappings(mapping_paths)
        return self._mappings
    
    def _load_data(self) -> Dict[str, List[pd.DataFrame]]:
        """Load all DataFrames for all splits."""
        if self._dataframes:
            return self._dataframes
        
        mappings = self._load_mappings()
        
        for experiment in self.config.experiments:
            split_name = experiment.split_name
            dfs = []
            
            for npy_path in experiment.npy_paths:
                logger.info(f"Loading {npy_path}")
                path = Path(npy_path)
                if self.config.base_dir:
                    path = Path(self.config.base_dir) / path
                
                try:
                    evaluation = self.loader.load_evaluation(path)
                    df = make_dataframe(evaluation, mappings)
                    dfs.append(df)
                except FileNotFoundError:
                    logger.warning(f"File not found: {path}")
                    continue
            
            if dfs:
                self._dataframes[split_name] = dfs
                logger.info(f"Loaded {len(dfs)} DataFrames for split '{split_name}'")
        
        return self._dataframes
    
    def run(
        self, 
        plots: List[str] = None, 
        save_csv: bool = True
    ):
        """
        Execute the visualization pipeline.
        
        Args:
            plots: List of plot types to generate (default: all)
            save_csv: Whether to save CSV data files
        """
        plots = plots or PLOT_TYPES
        
        if not self.dry_run:
            self.saver.ensure_dir()
        
        data = self._load_data()
        
        if not data:
            logger.error("No data loaded. Check your configuration paths.")
            return
        
        for plot_type in plots:
            try:
                logger.info(f"Generating {plot_type} plots...")
                
                if plot_type == "error_rate":
                    self._generate_error_rate_plots(data, save_csv)
                elif plot_type == "accuracy":
                    self._generate_accuracy_plots(data, save_csv)
                elif plot_type == "precision_recall":
                    self._generate_pr_scatter_plots(data, save_csv)
                elif plot_type == "pr_curve":
                    self._generate_pr_curve_plots(data)
                elif plot_type == "roc_curve":
                    self._generate_roc_curve_plots(data)
                elif plot_type == "confusion_matrix":
                    self._generate_confusion_matrix_plots(data)
                elif plot_type == "confidence":
                    self._generate_confidence_plots(data)
                elif plot_type in ("umap", "tsne"):
                    self._generate_dimensionality_plots(data, plot_type)
                else:
                    logger.warning(f"Unknown plot type: {plot_type}")
                    
            except Exception as e:
                logger.error(f"Error generating {plot_type} plots: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception("Full traceback:")
        
        # Generate and save report if requested
        if self.report and not self.dry_run:
            report_path = self.report.save(
                filename=self._timestamped_filename("report.md")
            )
            logger.info(f"Report saved to {report_path}")
    
    def _track_figure(self, filename: str):
        """Track a generated figure for the report."""
        if self.report:
            self.report.add_figure(filename)
    
    def _timestamped_filename(self, base_filename: str) -> str:
        """Add run timestamp prefix to filename."""
        return f"{self._run_timestamp}_{base_filename}"
    
    def _generate_error_rate_plots(
        self,
        data: Dict[str, List[pd.DataFrame]],
        save_csv: bool
    ):
        """Generate error rate plots for all splits."""
        error_rate_columns = ["labels", "well_ids", "plate_ids"]

        for split in self.config.splits:
            if split not in data:
                logger.warning(f"No data for split '{split}'")
                continue

            dfs = data[split]

            # Build well→label map once per split (for well_ids coloring)
            well_label_map = {}
            for df in dfs:
                well_label_map.update(dict(zip(df["well_ids"], df["labels"])))

            for column in error_rate_columns:
                # Calculate statistics
                stats = ErrorCalculator.aggregate_errors(dfs, column)

                # Generate figure
                fig = self.error_plotter.plot(
                    stats=stats,
                    column=column,
                    split=split,
                    well_label_map=well_label_map if column == "well_ids" else None,
                )

                # Save
                if not self.dry_run:
                    filename = self._timestamped_filename(f"{split}_error_rate_per_{column}.png")
                    self.saver.save_figure(
                        fig,
                        filename,
                        dpi=self.config.plot_settings.dpi
                    )
                    self._track_figure(filename)

                    if save_csv:
                        self.saver.save_dataframe(
                            stats.to_dataframe(),
                            self._timestamped_filename(f"{split}_error_rate_per_{column}.csv")
                        )
                        self.saver.save_dataframe(
                            stats.get_replicate_dataframe(column),
                            self._timestamped_filename(f"{split}_error_rate_replicate_means_per_{column}.csv")
                        )
                plt.close(fig)
    
    def _generate_accuracy_plots(
        self, 
        data: Dict[str, List[pd.DataFrame]], 
        save_csv: bool
    ):
        """Generate accuracy comparison plots."""
        # Filter to configured splits
        filtered_data = {
            k: v for k, v in data.items() 
            if k in self.config.splits
        }
        
        if not filtered_data:
            logger.warning("No data available for configured splits")
            return
        
        per_run = compute_split_accuracies(filtered_data)
        summary = summarize_accuracies(per_run, order=self.config.splits)
        
        # Create accuracy bar plot
        fig = self.acc_plotter.plot_accuracy_bars(
            summary, 
            order=self.config.splits
        )
        
        if not self.dry_run:
            filename = self._timestamped_filename("Test_Accuracy.png")
            self.saver.save_figure(
                fig, 
                filename,
                dpi=self.config.plot_settings.dpi
            )
            self._track_figure(filename)
            
            if save_csv:
                self.saver.save_dataframe(per_run, self._timestamped_filename("Test_Accuracy_per_run.csv"))
                self.saver.save_dataframe(summary, self._timestamped_filename("Test_Accuracy_summary.csv"))
    
    def _generate_pr_scatter_plots(
        self, 
        data: Dict[str, List[pd.DataFrame]], 
        save_csv: bool
    ):
        """Generate precision-recall scatter plots."""
        all_pr_dfs = []
        
        for split in self.config.splits:
            if split not in data:
                continue
            
            pr_df = PrecisionRecallCalculator.aggregate_across_runs(
                data[split], split
            )
            all_pr_dfs.append(pr_df)
            
            if save_csv and not self.dry_run:
                self.saver.save_dataframe(
                    pr_df,
                    self._timestamped_filename(f"{split}_precision_recall_per_class.csv")
                )
        
        if all_pr_dfs:
            full_pr_df = pd.concat(all_pr_dfs, ignore_index=True)
            
            fig = self.pr_plotter.plot_scatter(
                full_pr_df,
                self.config.architecture
            )
            
            if not self.dry_run:
                filename = self._timestamped_filename(f"{self.config.architecture}_PR_scatter_all_labels.png")
                self.saver.save_figure(
                    fig,
                    filename,
                    dpi=self.config.plot_settings.dpi
                )
                self._track_figure(filename)
    
    def _generate_pr_curve_plots(self, data: Dict[str, List[pd.DataFrame]]):
        """Generate PR curve plots with AUC."""
        curves = {}
        
        for split in self.config.splits:
            if split not in data:
                continue
            
            # Use first replicate for the curve
            curve, aucs = PrecisionRecallCalculator.aggregate_curves_across_runs(
                data[split], curve_type="pr"
            )
            curves[split] = curve
            logger.info(f"  {split} PR-AUC: {sum(aucs)/len(aucs):.3f}")
        
        if curves:
            fig = self.pr_plotter.plot_pr_curves(curves, self.config.architecture)
            
            if not self.dry_run:
                filename = self._timestamped_filename(f"{self.config.architecture}_PR_curve_auc.png")
                self.saver.save_figure(
                    fig,
                    filename,
                    dpi=self.config.plot_settings.dpi
                )
                self._track_figure(filename)
    
    def _generate_roc_curve_plots(self, data: Dict[str, List[pd.DataFrame]]):
        """Generate ROC curve plots with AUC."""
        curves = {}
        
        for split in self.config.splits:
            if split not in data:
                continue
            
            curve, aucs = PrecisionRecallCalculator.aggregate_curves_across_runs(
                data[split], curve_type="roc"
            )
            curves[split] = curve
            logger.info(f"  {split} ROC-AUC: {sum(aucs)/len(aucs):.3f}")
        
        if curves:
            fig = self.pr_plotter.plot_roc_curves(curves, self.config.architecture)
            
            if not self.dry_run:
                filename = self._timestamped_filename(f"{self.config.architecture}_ROC_curve_auc.png")
                self.saver.save_figure(
                    fig,
                    filename,
                    dpi=self.config.plot_settings.dpi
                )
                self._track_figure(filename)
    
    def _generate_confusion_matrix_plots(self, data: Dict[str, List[pd.DataFrame]]):
        """Generate confusion matrix plots."""
        for split in self.config.splits:
            if split not in data:
                continue

            combined_df = pd.concat(data[split], ignore_index=True)

            fig = self.cm_plotter.plot_confusion_matrix(
                combined_df,
                architecture=self.config.architecture,
                split=split,
            )

            if not self.dry_run:
                filename = self._timestamped_filename(f"{split}_confusion_matrix.png")
                self.saver.save_figure(
                    fig,
                    filename,
                    dpi=self.config.plot_settings.dpi
                )
                self._track_figure(filename)

    def _generate_confidence_plots(self, data: Dict[str, List[pd.DataFrame]]):
        """Generate confidence distribution plots."""
        for split in self.config.splits:
            if split not in data:
                continue
            
            # Combine all replicates
            combined_df = pd.concat(data[split], ignore_index=True)
            
            fig = self.dist_plotter.plot_confidence(
                combined_df,
                architecture=self.config.architecture,
                split=split
            )
            
            if not self.dry_run:
                filename = self._timestamped_filename(f"{split}_confidence_distribution.png")
                self.saver.save_figure(
                    fig,
                    filename,
                    dpi=self.config.plot_settings.dpi
                )
                self._track_figure(filename)
    
    def _generate_dimensionality_plots(
        self, 
        data: Dict[str, List[pd.DataFrame]],
        method: str
    ):
        """Generate UMAP or t-SNE plots."""
        for split in self.config.splits:
            if split not in data:
                continue
            
            # Combine all replicates
            combined_df = pd.concat(data[split], ignore_index=True)
            
            try:
                if method == "umap":
                    fig = self.dim_plotter.plot_umap(
                        combined_df,
                        color_by="labels",
                        title=f"{self.config.architecture} - UMAP by Label: {split}"
                    )
                else:
                    fig = self.dim_plotter.plot_tsne(
                        combined_df,
                        title=f"{self.config.architecture} - t-SNE: {split}"
                    )
                
                if not self.dry_run:
                    filename = self._timestamped_filename(f"{split}_{method}_labels.png")
                    self.saver.save_figure(
                        fig,
                        filename,
                        dpi=self.config.plot_settings.dpi
                    )
                    self._track_figure(filename)
            except ImportError as e:
                logger.warning(f"Skipping {method} plot: {e}")
            except Exception as e:
                logger.error(f"Error generating {method} plot for {split}: {e}")
