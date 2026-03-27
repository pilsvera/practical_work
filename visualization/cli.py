"""
Command-line interface for the visualization pipeline.

Usage:
    python -m masterthesis.visualization --config configs/source_3.yaml
    python -m masterthesis.visualization --config configs/source_3.yaml --plots error_rate accuracy
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from .config import VisualizationConfig
from .pipeline import VisualizationPipeline, PLOT_TYPES

logger = logging.getLogger(__name__)


def parse_args(args: List[str] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: List of arguments (uses sys.argv if None)
        
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate evaluation visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (overrides config)"
    )
    
    parser.add_argument(
        "--plots", "-p",
        nargs="+",
        choices=PLOT_TYPES,
        default=None,
        help=f"Specific plots to generate. Available: {', '.join(PLOT_TYPES)}"
    )
    
    parser.add_argument(
        "--splits", "-s",
        nargs="+",
        default=None,
        help="Specific splits to process (default: all from config)"
    )
    
    parser.add_argument(
        "--no-save-csv",
        action="store_true",
        help="Skip saving CSV data files"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without saving files"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating markdown documentation report"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)"
    )
    
    return parser.parse_args(args)


def setup_logging(verbosity: int):
    """
    Configure logging based on verbosity level.
    
    Args:
        verbosity: 0=WARNING, 1=INFO, 2+=DEBUG
    """
    level = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }.get(verbosity, logging.DEBUG)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )


def main(args: List[str] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command-line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success)
    """
    parsed_args = parse_args(args)
    setup_logging(parsed_args.verbose)
    
    # Load configuration
    logger.info(f"Loading config from {parsed_args.config}")
    try:
        config = VisualizationConfig.from_yaml(parsed_args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {parsed_args.config}")
        return 1
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return 1
    
    # Override config with CLI args
    if parsed_args.output_dir:
        config.output_dir = str(parsed_args.output_dir)
    if parsed_args.splits:
        config.splits = parsed_args.splits
    
    # Run pipeline
    logger.info(f"Output directory: {config.get_output_dir()}")
    logger.info(f"Splits: {config.splits}")
    
    if parsed_args.dry_run:
        logger.info("DRY RUN - no files will be saved")
    
    pipeline = VisualizationPipeline(
        config, 
        dry_run=parsed_args.dry_run,
        generate_report=not parsed_args.no_report
    )
    pipeline.run(
        plots=parsed_args.plots or PLOT_TYPES,
        save_csv=not parsed_args.no_save_csv
    )
    
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
