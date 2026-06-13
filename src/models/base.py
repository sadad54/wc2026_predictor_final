"""
src/models/base.py

Abstract base class that all outcome models must implement.

Every model in the ensemble honours the same interface:
  fit(X, y)          → trains the model
  predict_proba(X)   → returns (n, 3) matrix: [P(HW), P(D), P(AW)]
  predict(X)         → returns argmax of predict_proba
  save / load        → joblib serialisation

Because all models share this interface, the stacking ensemble
can treat them interchangeably — swapping in a new model is a
one-line change, not a rewrite.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

class BaseOutcomeModel(ABC):
    """
    Abstract base for match outcome probability models.
    All implementations must predict [P(home win), P(draw), P(away win)].
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "BaseOutcomeModel":
        """Train the model on feature matrix X and labels y."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return (n_samples, 3) probability matrix.
        Column order: [P(home win), P(draw), P(away win)]
        Rows must sum to 1.0.
        """
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted outcome class: 0=home win, 1=draw, 2=away win."""
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path: Path) -> None:
        """Serialise model to disk with joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "BaseOutcomeModel":
        """Deserialise model from disk."""
        return joblib.load(path)