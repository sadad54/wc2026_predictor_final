"""
tests/test_utils.py

Unit tests for src/utils/helpers.py and src/utils/logger.py

Run with: pytest tests/test_utils.py -v
"""

import random
from pathlib import Path

import numpy as np
import pytest

from src.utils.helpers import (
    ensure_dir,
    get_project_root,
    load_config,
    set_random_seeds,
)


# ─────────────────────────────────────────────────────────────────────────────
# load_config tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadConfig:

    def test_returns_dictionary(self):
        config = load_config()
        assert isinstance(config, dict)

    def test_has_required_top_level_keys(self):
        config = load_config()
        required = {"project", "paths", "data", "features", "models", "simulation", "logging"}
        assert required.issubset(set(config.keys()))

    def test_project_name_correct(self):
        config = load_config()
        assert config["project"]["name"] == "worldcup-2026-predictor"

    def test_simulation_count(self):
        config = load_config()
        assert config["simulation"]["n_simulations"] == 10000

    def test_elo_initial_rating(self):
        config = load_config()
        assert config["features"]["elo"]["initial_rating"] == 1500

    def test_n_teams(self):
        config = load_config()
        assert config["data"]["tournament"]["n_teams"] == 48

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("config/does_not_exist.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# ensure_dir tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureDir:

    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "subdir" / "nested"
        result = ensure_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_returns_path_object(self, tmp_path):
        result = ensure_dir(tmp_path / "test")
        assert isinstance(result, Path)

    def test_safe_on_existing_directory(self, tmp_path):
        """Should not raise if directory already exists."""
        ensure_dir(tmp_path)    # tmp_path already exists
        ensure_dir(tmp_path)    # calling again should be safe

    def test_accepts_string_input(self, tmp_path):
        result = ensure_dir(str(tmp_path / "string_path"))
        assert result.exists()


# ─────────────────────────────────────────────────────────────────────────────
# get_project_root tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetProjectRoot:

    def test_returns_path_object(self):
        root = get_project_root()
        assert isinstance(root, Path)

    def test_config_exists_at_root(self):
        root = get_project_root()
        assert (root / "config" / "config.yaml").exists()

    def test_src_exists_at_root(self):
        root = get_project_root()
        assert (root / "src").exists()


# ─────────────────────────────────────────────────────────────────────────────
# set_random_seeds tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSetRandomSeeds:

    def test_numpy_reproducible(self):
        set_random_seeds(42)
        a = np.random.rand(5)

        set_random_seeds(42)
        b = np.random.rand(5)

        np.testing.assert_array_equal(a, b)

    def test_python_random_reproducible(self):
        set_random_seeds(42)
        a = [random.random() for _ in range(5)]

        set_random_seeds(42)
        b = [random.random() for _ in range(5)]

        assert a == b

    def test_different_seeds_give_different_results(self):
        set_random_seeds(42)
        a = np.random.rand(5)

        set_random_seeds(99)
        b = np.random.rand(5)

        assert not np.array_equal(a, b)

    def test_does_not_raise(self):
        """Should complete without errors even with unusual seed values."""
        set_random_seeds(0)
        set_random_seeds(1)
        set_random_seeds(999999)