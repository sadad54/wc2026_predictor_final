"""
src/models/__init__.py
Clean public interface for all model classes.
"""

from src.models.dixon_coles import DixonColesModel
from src.models.ensemble import StackingEnsemble
from src.models.outcome_models import (
    FEATURE_COLS,
    EloOutcomeModel,
    LogisticOutcomeModel,
    RandomForestOutcomeModel,
    XGBoostOutcomeModel,
)

__all__ = [
    "DixonColesModel",
    "StackingEnsemble",
    "EloOutcomeModel",
    "XGBoostOutcomeModel",
    "RandomForestOutcomeModel",
    "LogisticOutcomeModel",
    "FEATURE_COLS",
]