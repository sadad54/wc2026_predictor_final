"""
validate_setup.py

Health check script for Phase 0.
Run from the project root: python validate_setup.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

import importlib
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# What we check
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = {
    "pandas":       "pandas",
    "numpy":        "numpy",
    "scipy":        "scipy",
    "sklearn":      "scikit-learn",
    "xgboost":      "xgboost",
    "lightgbm":     "lightgbm",
    "statsmodels":  "statsmodels",
    "shap":         "shap",
    "loguru":       "loguru",
    "yaml":         "pyyaml",
    "dotenv":       "python-dotenv",
    "plotly":       "plotly",
    "matplotlib":   "matplotlib",
    "seaborn":      "seaborn",
    "tqdm":         "tqdm",
    "joblib":       "joblib",
    "requests":     "requests",
    "bs4":          "beautifulsoup4",
    "pytest":       "pytest",
}

REQUIRED_DIRS = [
    "config",
    "data/raw",
    "data/processed",
    "data/external",
    "logs",
    "models/saved",
    "models/metrics",
    "notebooks",
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/simulation",
    "src/utils",
    "tests",
]

REQUIRED_FILES = [
    "config/config.yaml",
    "src/__init__.py",
    "src/utils/__init__.py",
    "src/utils/helpers.py",
    "src/utils/logger.py",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    ".env.example",
]

# ─────────────────────────────────────────────────────────────────────────────
# Check functions
# ─────────────────────────────────────────────────────────────────────────────

def _pass(msg: str) -> None:
    print(f"  [OK]  {msg}")

def _fail(msg: str) -> None:
    print(f"  [FAIL]  {msg}")


def check_python_version() -> int:
    """Verify Python 3.11.x is active."""
    v = sys.version_info
    if v.major == 3 and v.minor == 11:
        _pass(f"Python {v.major}.{v.minor}.{v.micro}")
        return 0
    else:
        _fail(f"Python {v.major}.{v.minor}.{v.micro} — expected 3.11.x")
        return 1


def check_packages() -> int:
    """Verify all required packages are importable."""
    failures = 0
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            _pass(f"{pip_name} ({version})")
        except ImportError:
            _fail(f"{pip_name} — not installed (run: pip install {pip_name})")
            failures += 1
    return failures


def check_directories() -> int:
    """Verify all required directories exist."""
    failures = 0
    for dir_path in REQUIRED_DIRS:
        if Path(dir_path).is_dir():
            _pass(f"{dir_path}/")
        else:
            _fail(f"{dir_path}/ — missing")
            failures += 1
    return failures


def check_files() -> int:
    """Verify all required files exist."""
    failures = 0
    for file_path in REQUIRED_FILES:
        if Path(file_path).is_file():
            _pass(file_path)
        else:
            _fail(f"{file_path} — missing")
            failures += 1
    return failures


def check_config_loads() -> int:
    """Verify config.yaml is valid YAML and has the expected top-level keys."""
    try:
        import yaml
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        expected_keys = {"project", "paths", "data", "features", "models", "simulation", "logging"}
        missing = expected_keys - set(config.keys())
        if missing:
            _fail(f"config.yaml missing keys: {missing}")
            return 1
        _pass(f"config.yaml loaded — {len(config)} top-level sections")
        return 0
    except Exception as e:
        _fail(f"config.yaml failed to load — {e}")
        return 1


def check_logger_initialises() -> int:
    """Verify the logger can be set up without errors."""
    try:
        from src.utils.logger import setup_logger
        setup_logger(log_dir="logs", level="INFO")
        _pass("Logger initialised (console + file handlers)")
        return 0
    except Exception as e:
        _fail(f"Logger failed to initialise — {e}")
        return 1


def check_initialize_project() -> int:
    """End-to-end check: run the full project bootstrap."""
    try:
        helpers = importlib.import_module("src.utils.helpers")
        initialize_project = getattr(helpers, "initialize_project")
        config = initialize_project()
        assert config["project"]["name"] == "worldcup-2026-predictor"
        assert config["simulation"]["n_simulations"] == 10000
        _pass("initialize_project() ran successfully")
        return 0
    except Exception as e:
        _fail(f"initialize_project() failed — {e}")
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    width = 60
    print("\n" + "=" * width)
    print(" WORLD CUP PREDICTOR - PHASE 0 VALIDATION")
    print("=" * width)

    sections = [
        ("Python version",    check_python_version),
        ("Required packages", check_packages),
        ("Directory structure", check_directories),
        ("Required files",    check_files),
        ("Config loads",      check_config_loads),
        ("Logger initialises", check_logger_initialises),
        ("Full bootstrap",    check_initialize_project),
    ]

    total_failures = 0
    for section_name, check_fn in sections:
        print(f"\n[{section_name}]")
        total_failures += check_fn()

    print("\n" + "=" * width)
    if total_failures == 0:
        print("  [OK]  ALL CHECKS PASSED - Phase 0 complete!")
    else:
        print(f"  [FAIL]  {total_failures} CHECK(S) FAILED - fix the issues above")
    print("=" * width + "\n")

    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()