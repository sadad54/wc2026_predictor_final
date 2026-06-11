"""
src/utils/__init__.py

Re-exports the most-used utilities so other modules can do:
    from src.utils import logger, load_config, initialize_project
instead of the longer:
    from src.utils.helpers import load_config
    from src.utils.logger import logger
"""

from . import helpers as _helpers

# Import attributes from helpers via getattr to avoid static "No name" checks
ensure_dir = getattr(_helpers, "ensure_dir")
get_project_root = getattr(_helpers, "get_project_root")
initialize_project = getattr(_helpers, "initialize_project")
load_config = getattr(_helpers, "load_config")
set_random_seeds = getattr(_helpers, "set_random_seeds")
from src.utils.logger import logger, setup_logger

__all__ = [
    "load_config",
    "ensure_dir",
    "get_project_root",
    "set_random_seeds",
    "initialize_project",
    "logger",
    "setup_logger",
]