"""
Base Model Interface
--------------------
All models implement this interface for consistent usage.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseGDPModel(ABC):
    """Abstract base class for GDP estimation models."""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs,
    ) -> dict:
        """
        Train the model with hyperparameter optimization.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input X."""
        pass

    def predict_monthly(self, X_m_scaled: np.ndarray) -> np.ndarray:
        """
        Generate monthly signal. Override for models (e.g. MLP)
        that require special input formatting.
        """
        return self.predict(X_m_scaled)

    def get_name(self) -> str:
        return self.name
