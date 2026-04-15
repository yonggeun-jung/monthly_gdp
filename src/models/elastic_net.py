"""
Elastic Net Model
-----------------
Regularized linear regression with bootstrap inference.
"""

import numpy as np
from tqdm import tqdm
from .base import BaseGDPModel


class ElasticNetModel(BaseGDPModel):
    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(name="ElasticNet", seed=seed)
        self.config = {
            "n_bootstrap": kwargs.get("n_bootstrap", 5000),
            "n_alphas": kwargs.get("n_alphas", 100),
            "cv_folds": kwargs.get("cv_folds", 5),
        }
        self.beta_hat = None
        self.intercept = None
        self.bootstrap_betas = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        from sklearn.linear_model import ElasticNetCV

        np.random.seed(self.seed)

        # Step 1: find optimal lambda_1, lambda_2 via CV
        enet_cv = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
            n_alphas=self.config["n_alphas"],
            cv=self.config["cv_folds"],
            random_state=self.seed,
            max_iter=100000,
        )
        enet_cv.fit(X_train, y_train)
        best_alpha = enet_cv.alpha_
        best_l1_ratio = enet_cv.l1_ratio_

        # Step 2: bootstrap
        from sklearn.linear_model import ElasticNet as EN

        B = self.config["n_bootstrap"]
        n = len(X_train)
        beta_list = []

        for _ in tqdm(range(B), desc="Bootstrap", disable=kwargs.get("quiet", False)):
            idx = np.random.choice(n, size=n, replace=True)
            model_b = EN(
                alpha=best_alpha,
                l1_ratio=best_l1_ratio,
                max_iter=100000,
            )
            model_b.fit(X_train[idx], y_train[idx])
            beta_list.append(model_b.coef_)

        self.bootstrap_betas = np.array(beta_list)
        self.beta_hat = np.mean(self.bootstrap_betas, axis=0)
        self.intercept = np.mean(y_train) - X_train.mean(axis=0) @ self.beta_hat

        # Store the CV model for reference
        self.model = enet_cv
        self.is_fitted = True

        return {
            "alpha": best_alpha,
            "l1_ratio": best_l1_ratio,
            "n_bootstrap": B,
        }

    def predict(self, X):
        return (X @ self.beta_hat + self.intercept).flatten()

    def fit_fixed(self, X_train, y_train, X_val=None, y_val=None, hp_dict=None, **kwargs):
        """
        Re-estimate with fixed (alpha, l1_ratio) from the first window.
        Skips CV search and bootstrap for speed during expanding-window
        evaluation.  Bootstrap should be run only on the final full sample.
        """
        from sklearn.linear_model import ElasticNet as EN

        if hp_dict is None or "alpha" not in hp_dict or "l1_ratio" not in hp_dict:
            raise ValueError("hp_dict must contain 'alpha' and 'l1_ratio' for fit_fixed().")

        np.random.seed(self.seed)
        model = EN(
            alpha=hp_dict["alpha"],
            l1_ratio=hp_dict["l1_ratio"],
            max_iter=100000,
        )
        model.fit(X_train, y_train)
        self.beta_hat = model.coef_
        self.intercept = model.intercept_
        self.model = model
        self.is_fitted = True
        self.bootstrap_betas = None  # not available in fixed mode
        return hp_dict

    def get_coefficients(self, feature_names: list) -> dict:
        """Return bootstrap-averaged coefficients."""
        return dict(zip(feature_names, self.beta_hat))

    def get_confidence_intervals(self, X_m, alpha: float = 0.05):
        """Compute bootstrap confidence intervals for monthly predictions."""
        if self.bootstrap_betas is None:
            raise ValueError(
                "Bootstrap coefficients are unavailable. "
                "Run fit() with n_bootstrap > 0 before requesting confidence intervals."
            )
        preds = X_m @ self.bootstrap_betas.T + self.intercept
        lower = np.percentile(preds, 100 * alpha / 2, axis=1)
        upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=1)
        return lower, upper