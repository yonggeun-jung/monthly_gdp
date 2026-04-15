"""
XGBoost Model
-------------
Gradient boosting with grid search hyperparameter optimization.
"""

import numpy as np
from .base import BaseGDPModel


class XGBoostModel(BaseGDPModel):
    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(name="XGBoost", seed=seed)
        self.config = {
            "cv_folds": kwargs.get("cv_folds", 5),
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.05, 0.1, 0.3],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "gamma": [0, 0.1],
            "min_child_weight": [5, 10],
        }
        self.best_hp = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            "n_estimators": self.config["n_estimators"],
            "max_depth": self.config["max_depth"],
            "learning_rate": self.config["learning_rate"],
            "subsample": self.config["subsample"],
            "colsample_bytree": self.config["colsample_bytree"],
            "gamma": self.config["gamma"],
            "min_child_weight": self.config["min_child_weight"],
        }

        base = xgb.XGBRegressor(
            objective="reg:squarederror", random_state=self.seed
        )
        grid = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=self.config["cv_folds"],
            verbose=0,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        self.best_hp = grid.best_params_
        self.is_fitted = True
        return self.best_hp

    def fit_fixed(self, X_train, y_train, X_val=None, y_val=None, hp_dict=None, **kwargs):
        """Train with fixed hyperparameters (no grid search)."""
        import xgboost as xgb

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=self.seed,
            **hp_dict,
        )
        model.fit(X_train, y_train)
        self.model = model
        self.best_hp = hp_dict
        self.is_fitted = True
        return hp_dict

    def predict(self, X):
        return self.model.predict(X).flatten()

    def get_feature_importance(self, feature_names: list) -> dict:
        """Return feature importance scores (gain-based)."""
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        ))
