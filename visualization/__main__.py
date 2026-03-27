"""
Entry point for running the visualization package as a module.

Usage:
    python -m visualization --config visualization/configs/source_3.yaml
"""

from .cli import main

if __name__ == "__main__":
    main()
