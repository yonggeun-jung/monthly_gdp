"""
Chow-Lin Temporal Disaggregation (Non-ML Benchmark)
----------------------------------------------------
Classical regression-based method (Chow and Lin, 1971).

Estimates beta by GLS with AR(1) residual structure, then
distributes quarterly GDP to months using the estimated
relationship and monthly indicator movements.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from .base import BaseGDPModel


class ChowLinModel(BaseGDPModel):
    """
    Chow-Lin disaggregation with AR(1) residuals.

    Model: Y_q = X_q beta + u_q
    where u_q has covariance induced by AR(1) monthly errors.

    The optimal rho is found by maximizing the GLS log-likelihood.
    """

    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(name="ChowLin", seed=seed)
        self.beta = None
        self.rho = None
        self._ols = None

    def _build_ar1_cov(self, T: int, rho: float) -> np.ndarray:
        """Build T x T AR(1) covariance matrix."""
        V = np.zeros((T, T))
        for i in range(T):
            for j in range(T):
                V[i, j] = rho ** abs(i - j)
        return V / (1 - rho ** 2)

    def _aggregate_matrix(self, T_monthly: int) -> np.ndarray:
        """
        Build C matrix that aggregates monthly to quarterly.
        C is Q x T_monthly, each row sums 3 consecutive months.
        """
        Q = T_monthly // 3
        C = np.zeros((Q, T_monthly))
        for q in range(Q):
            C[q, 3 * q: 3 * q + 3] = 1.0
        return C

    def _gls_loglik(self, rho: float, Y_q, X_q, C, T_monthly):
        """Negative GLS log-likelihood as function of rho."""
        if abs(rho) >= 0.999:
            return 1e12
        V_m = self._build_ar1_cov(T_monthly, rho)
        V_q = C @ V_m @ C.T

        V_q_inv = np.linalg.pinv(V_q)

        # GLS beta
        XtVi = X_q.T @ V_q_inv
        beta = np.linalg.lstsq(XtVi @ X_q, XtVi @ Y_q, rcond=None)[0]
        resid = Y_q - X_q @ beta

        # Log-likelihood (up to constant)
        sign, logdet = np.linalg.slogdet(V_q)
        if sign <= 0:
            return 1e12
        ll = -0.5 * (logdet + resid.T @ V_q_inv @ resid)
        return -ll  # negative for minimization

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Estimate beta and rho via GLS.

        Note: X_train and y_train are quarterly-level data.
        """
        Q = len(y_train)
        T_monthly = Q * 3

        Y_q = y_train.flatten()
        X_q = X_train.copy()
        if X_q.ndim == 1:
            X_q = X_q.reshape(-1, 1)

        C = self._aggregate_matrix(T_monthly)

        # Find optimal rho
        result = minimize_scalar(
            self._gls_loglik,
            bounds=(-0.99, 0.99),
            method="bounded",
            args=(Y_q, X_q, C, T_monthly),
        )
        self.rho = result.x

        # Final GLS estimation with optimal rho
        V_m = self._build_ar1_cov(T_monthly, self.rho)
        V_q = C @ V_m @ C.T
        V_q_inv = np.linalg.pinv(V_q)

        XtVi = X_q.T @ V_q_inv
        self.beta = np.linalg.lstsq(XtVi @ X_q, XtVi @ Y_q, rcond=None)[0]

        # Also store OLS for comparison
        self._ols = LinearRegression(fit_intercept=False)
        self._ols.fit(X_q, Y_q)

        self.model = True
        self.is_fitted = True

        return {"rho": self.rho, "n_features": X_q.shape[1]}

    def predict(self, X):
        """Predict quarterly growth using GLS coefficients."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X @ self.beta).flatten()

    def disaggregate(self, X_q, Y_q, X_m):
        """
        Full Chow-Lin disaggregation to monthly frequency.

        Parameters
        ----------
        X_q : ndarray, shape (Q, k)
            Quarterly explanatory variables.
        Y_q : ndarray, shape (Q,)
            Observed quarterly GDP growth.
        X_m : ndarray, shape (T, k)
            Monthly explanatory variables.

        Returns
        -------
        y_m : ndarray, shape (T,)
            Estimated monthly GDP growth.
        """
        Q = len(Y_q)
        T = Q * 3
        C = self._aggregate_matrix(T)

        V_m = self._build_ar1_cov(T, self.rho)
        V_q = C @ V_m @ C.T
        V_q_inv = np.linalg.pinv(V_q)

        # Monthly preliminary estimate
        y_m_hat = X_m @ self.beta

        # Quarterly residual
        y_q_hat = C @ y_m_hat
        u_q = Y_q - y_q_hat

        # Distribute residual: y_m = y_m_hat + V_m C' V_q^{-1} u_q
        y_m = y_m_hat + V_m @ C.T @ V_q_inv @ u_q

        return y_m
