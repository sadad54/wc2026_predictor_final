"""
src/data/download.py

Phase 1: Download all raw data from Kaggle.

Downloads two datasets:
  1. International match results (1872–present)
  2. FIFA World Rankings history (1992–present)

Run from the project root:
    python -m src.data.download

Outputs (in data/raw/):
  - results.csv
  - rankings.csv
"""

import importlib
import os
from pathlib import Path

from dotenv import load_dotenv

from src.utils import initialize_project, logger


def download_kaggle_dataset(dataset_slug: str, dest_dir: Path) -> None:
    """
    Download a Kaggle dataset and extract it to dest_dir.

    Args:
        dataset_slug: The Kaggle dataset identifier, e.g. 'martj42/international-football-results'
        dest_dir: Directory to extract files into.
    """
    try:
        kaggle = importlib.import_module("kaggle")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'kaggle' library is required to download datasets. "
            "Install it with 'pip install kaggle' and ensure your environment can resolve it."
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading: {dataset_slug}")
    kaggle.api.dataset_download_files(dataset_slug, path=str(dest_dir), unzip=True)
    logger.info(f"Saved to: {dest_dir}")


def download_all(config: dict) -> None:
    """Download all raw datasets defined in config."""
    load_dotenv()

    # Set Kaggle credentials from .env
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY", "")

    if not os.environ["KAGGLE_USERNAME"] or not os.environ["KAGGLE_KEY"]:
        raise EnvironmentError(
            "KAGGLE_USERNAME or KAGGLE_KEY not set. "
            "Copy .env.example to .env and fill in your credentials."
        )

    raw_dir = Path(config["paths"]["raw_data"])
    kaggle_cfg = config["data"]["kaggle"]

    # ── 1. Match results ─────────────────────────────────────────────────────
    download_kaggle_dataset(kaggle_cfg["results_dataset"], raw_dir)

    # ── 2. FIFA rankings ─────────────────────────────────────────────────────
    download_kaggle_dataset(kaggle_cfg["rankings_dataset"], raw_dir)

    # ── Rename to consistent filenames ───────────────────────────────────────
    # The Kaggle datasets may extract with different names depending on version.
    # We standardise here so all downstream code uses predictable filenames.
    _rename_if_exists(raw_dir, "results.csv", "results.csv")
    _rename_if_exists(raw_dir, "fifa_ranking_2023-07-20.csv", "rankings.csv")
    _rename_if_exists(raw_dir, "fifa_ranking.csv", "rankings.csv")

    logger.info("Phase 1 download complete. Files in data/raw/:")
    for f in sorted(raw_dir.glob("*.csv")):
        size_mb = f.stat().st_size / 1_000_000
        logger.info(f"  {f.name}  ({size_mb:.1f} MB)")


def _rename_if_exists(directory: Path, old_name: str, new_name: str) -> None:
    """Rename a file in directory if old_name exists and new_name doesn't."""
    old = directory / old_name
    new = directory / new_name
    if old.exists() and not new.exists():
        old.rename(new)
        logger.debug(f"Renamed {old_name} → {new_name}")


if __name__ == "__main__":
    cfg = initialize_project()
    download_all(cfg)
