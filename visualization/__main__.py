"""
Entry point for running the visualization package as a module.

Usage:
    python -m masterthesis.visualization --config configs/source_3.yaml
"""

from .cli import main

if __name__ == "__main__":
    main()
