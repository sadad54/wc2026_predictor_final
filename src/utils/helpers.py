"""
src/utils/helpers.py

Shared utility functions for the World Cup Prediction Engine.
These are the building blocks used by every other module —
configuration loading, directory management, and reproducibility.
"""

import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Union[str, Path] = "config/config.yaml") -> Dict[str, Any]:
    """
    Load the central YAML configuration file.

    Args:
        config_path: Path to the config file. Defaults to 'config/config.yaml'.

    Returns:
        Dictionary containing all project configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file has invalid YAML syntax.

    Example:
        >>> config = load_config()
        >>> config["simulation"]["n_simulations"]
        10000
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. "
            "Make sure you are running from the project root directory."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ─────────────────────────────────────────────────────────────────────────────
# File System
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create a directory (and all parent directories) if it does not exist.
    Safe to call even if the directory already exists.

    Args:
        path: Directory path to create.

    Returns:
        Path object pointing to the (now definitely existing) directory.

    Example:
        >>> output_dir = ensure_dir("data/processed/features")
        >>> output_dir.exists()
        True
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    This works regardless of which subdirectory a script is run from,
    because it anchors to the location of this file (src/utils/helpers.py)
    and navigates two levels up.

    Returns:
        Path object for the project root.

    Example:
        >>> root = get_project_root()
        >>> (root / "config" / "config.yaml").exists()
        True
    """
    # This file is at: <root>/src/utils/helpers.py
    # .parent      →  src/utils/
    # .parent      →  src/
    # .parent      →  <root>/
    return Path(__file__).parent.parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds across all libraries used in the project to ensure
    fully reproducible results.

    In machine learning, any randomness (train/test splits, model
    initialisation, Monte Carlo sampling) will produce slightly different
    results each run unless the seed is fixed. This function locks them all.

    Args:
        seed: Integer seed value. Default is 42 (convention).

    Example:
        >>> set_random_seeds(42)
        >>> np.random.rand()   # will always return the same number
    """
    random.seed(seed)
    np.random.seed(seed)

    # XGBoost uses its own seed parameter (set in config), but we set numpy
    # so that any numpy-based operations inside XGBoost are reproducible too.

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch not installed — that's fine, we don't require it


# ─────────────────────────────────────────────────────────────────────────────
# Project Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def initialize_project(
    config_path: Union[str, Path] = "config/config.yaml",
) -> Dict[str, Any]:
    """
    One-call project bootstrap.

    Does four things in order:
      1. Loads the config file
      2. Creates all required directories (logs, data, models)
      3. Sets up the logger (console + rotating file)
      4. Sets all random seeds for reproducibility

    Call this at the top of every script and notebook in the project.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        The loaded configuration dictionary.

    Example:
        >>> config = initialize_project()
        >>> config["project"]["name"]
        'worldcup-2026-predictor'
    """
    config = load_config(config_path)

    # 1. Ensure all project directories exist
    for key, dir_path in config["paths"].items():
        ensure_dir(dir_path)

    # 2. Configure logging
    from src.utils.logger import setup_logger

    log_cfg = config["logging"]
    setup_logger(
        log_dir=config["paths"]["logs"],
        level=log_cfg["level"],
        rotation=log_cfg["rotation"],
        retention=log_cfg["retention"],
    )

    # 3. Set random seeds
    set_random_seeds(config["models"]["random_state"])

    # 4. Log startup message
    from src.utils.logger import logger

    logger.info(
        f"Project '{config['project']['name']}' "
        f"v{config['project']['version']} initialised | "
        f"seed={config['models']['random_state']}"
    )

    return config