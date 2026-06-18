"""
src/models/__init__.py
Clean public interface for all model classes.
"""

from src.models.dixon_coles import DixonColesModel

__all__ = [
    "DixonColesModel",
    "StackingEnsemble",
    "EloOutcomeModel",
    "XGBoostOutcomeModel",
    "RandomForestOutcomeModel",
    "LogisticOutcomeModel",
    "FEATURE_COLS",
]


def __getattr__(name: str):
    """Lazily import heavy optional model classes."""
    if name == "StackingEnsemble":
        from src.models.ensemble import StackingEnsemble

        return StackingEnsemble

    if name in {
        "FEATURE_COLS",
        "EloOutcomeModel",
        "LogisticOutcomeModel",
        "RandomForestOutcomeModel",
        "XGBoostOutcomeModel",
    }:
        from src.models import outcome_models

        return getattr(outcome_models, name)

    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
