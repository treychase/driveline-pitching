"""Bayesian Lasso regression via Gibbs sampling (pure NumPy).

This implements the Bayesian Lasso of Park & Casella (2008) using the
**scale-mixture-of-Gaussians** representation of the Laplace prior. Each
regression coefficient is given a *Gaussian prior* conditional on its own
variance,

    beta_j | sigma^2, tau_j^2  ~  Normal(0, sigma^2 * tau_j^2),

and the variances are mixed with an exponential hyper-prior,

    tau_j^2  ~  Exponential(lambda^2 / 2).

Marginalising over tau_j^2 yields the double-exponential (Laplace) prior that
produces Lasso-style shrinkage, while every full-conditional used by the Gibbs
sampler remains a tractable Gaussian / inverse-Gaussian / gamma draw. So the
model is a Bayesian Lasso whose coefficients have (conditional) Gaussian priors
— exactly the construction requested.

The sampler also places a Gamma hyper-prior on lambda^2 so the overall shrinkage
strength is learned from the data rather than tuned by hand.

Example
-------
    model = BayesianLasso().fit(X_train, y_train)
    mean, std = model.predict(X_test, return_std=True)
    summary = model.coef_summary(feature_names)   # posterior mean + 95% CI
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _rand_inverse_gaussian(mu: np.ndarray, lam: np.ndarray, rng) -> np.ndarray:
    """Sample from the Inverse-Gaussian(mu, lam) distribution (vectorised).

    Uses the Michael, Schucany & Haas (1976) transformation method.
    """
    nu = rng.standard_normal(mu.shape)
    y = nu ** 2
    x = (
        mu
        + (mu ** 2 * y) / (2.0 * lam)
        - (mu / (2.0 * lam)) * np.sqrt(4.0 * mu * lam * y + mu ** 2 * y ** 2)
    )
    z = rng.random(mu.shape)
    return np.where(z <= mu / (mu + x), x, mu ** 2 / x)


@dataclass
class BayesianLasso:
    """Bayesian Lasso linear regression fit by Gibbs sampling.

    Parameters
    ----------
    n_iter:
        Total number of Gibbs iterations.
    burn_in:
        Number of initial iterations discarded as warm-up.
    thin:
        Keep every ``thin``-th post-burn-in sample.
    lambda_shape, lambda_rate:
        Gamma(shape, rate) hyper-prior on the squared shrinkage parameter
        ``lambda^2``. The defaults are weakly informative.
    standardize:
        If True (default) features are centred and scaled internally, so the
        learned coefficients are directly comparable as importances and the
        Lasso penalty acts fairly across features of different units.
    random_state:
        Seed for reproducibility.
    """

    n_iter: int = 2000
    burn_in: int = 1000
    thin: int = 1
    lambda_shape: float = 1.0
    lambda_rate: float = 0.1
    standardize: bool = True
    random_state: int | None = 0

    # --- learned attributes (populated by .fit) -----------------------------
    def __post_init__(self) -> None:
        self.beta_samples_: np.ndarray | None = None   # (n_keep, p) standardized space
        self.sigma2_samples_: np.ndarray | None = None
        self.lambda2_samples_: np.ndarray | None = None
        self._x_mean = self._x_std = None
        self._y_mean = 0.0
        self.n_features_ = 0

    # ----------------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLasso":
        """Run the Gibbs sampler on training data ``X`` (n x p) and ``y`` (n)."""
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_ = p

        if self.standardize:
            self._x_mean = X.mean(axis=0)
            self._x_std = X.std(axis=0)
            self._x_std[self._x_std == 0] = 1.0
        else:
            self._x_mean = np.zeros(p)
            self._x_std = np.ones(p)
        Xs = (X - self._x_mean) / self._x_std
        self._y_mean = y.mean()
        yc = y - self._y_mean

        XtX = Xs.T @ Xs
        Xty = Xs.T @ yc

        # Initial values
        beta = np.linalg.lstsq(Xs, yc, rcond=None)[0]
        sigma2 = float(np.var(yc)) or 1.0
        inv_tau2 = np.ones(p)
        lambda2 = 1.0

        keep_beta, keep_sigma2, keep_lambda2 = [], [], []
        for it in range(self.n_iter):
            # --- beta | rest  ~  Normal(A^{-1} Xty, sigma2 A^{-1}) ---
            A = XtX + np.diag(inv_tau2)
            L = np.linalg.cholesky(A)
            mean = np.linalg.solve(A, Xty)
            z = rng.standard_normal(p)
            beta = mean + np.sqrt(sigma2) * np.linalg.solve(L.T, z)

            # --- sigma^2 | rest  ~  Inverse-Gamma ---
            resid = yc - Xs @ beta
            shape = (n - 1) / 2.0 + p / 2.0
            scale = 0.5 * (resid @ resid) + 0.5 * float(beta @ (inv_tau2 * beta))
            sigma2 = 1.0 / rng.gamma(shape, 1.0 / scale)

            # --- 1/tau_j^2 | rest  ~  Inverse-Gaussian ---
            mu_ig = np.sqrt(lambda2 * sigma2 / (beta ** 2))
            mu_ig = np.clip(mu_ig, 1e-8, 1e8)
            inv_tau2 = _rand_inverse_gaussian(mu_ig, lambda2 * np.ones(p), rng)
            inv_tau2 = np.clip(inv_tau2, 1e-8, 1e8)
            tau2 = 1.0 / inv_tau2

            # --- lambda^2 | rest  ~  Gamma (conjugate hyper-prior update) ---
            lambda2 = rng.gamma(
                p + self.lambda_shape,
                1.0 / (0.5 * tau2.sum() + self.lambda_rate),
            )

            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                keep_beta.append(beta.copy())
                keep_sigma2.append(sigma2)
                keep_lambda2.append(lambda2)

        self.beta_samples_ = np.asarray(keep_beta)
        self.sigma2_samples_ = np.asarray(keep_sigma2)
        self.lambda2_samples_ = np.asarray(keep_lambda2)
        return self

    # ----------------------------------------------------------------------- #
    @property
    def coef_(self) -> np.ndarray:
        """Posterior-mean coefficients in standardized feature space."""
        return self.beta_samples_.mean(axis=0)

    @property
    def intercept_(self) -> float:
        return float(self._y_mean)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self._x_mean) / self._x_std

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Posterior predictive mean (and optionally std) for new ``X``.

        The predictive std combines parameter uncertainty (spread of the
        posterior mean function) with observation noise ``sigma``.
        """
        Xs = self._standardize(X)
        # (n_keep, n_samples) matrix of fitted means across posterior draws
        fitted = self._y_mean + Xs @ self.beta_samples_.T
        mean = fitted.mean(axis=1)
        if not return_std:
            return mean
        param_var = fitted.var(axis=1)
        noise_var = self.sigma2_samples_.mean()
        std = np.sqrt(param_var + noise_var)
        return mean, std

    def coef_summary(self, feature_names=None, cred_mass: float = 0.95):
        """Return per-feature posterior mean, std, and credible interval.

        Returns a list of dicts (one per feature) sorted by absolute mean.
        """
        lo_q = (1 - cred_mass) / 2 * 100
        hi_q = 100 - lo_q
        means = self.beta_samples_.mean(axis=0)
        stds = self.beta_samples_.std(axis=0)
        los = np.percentile(self.beta_samples_, lo_q, axis=0)
        his = np.percentile(self.beta_samples_, hi_q, axis=0)
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i}" for i in range(self.n_features_)]
        )
        rows = [
            {
                "feature": names[i],
                "mean": float(means[i]),
                "std": float(stds[i]),
                "ci_low": float(los[i]),
                "ci_high": float(his[i]),
                # "selected" if the credible interval excludes 0
                "nonzero": bool(los[i] > 0 or his[i] < 0),
            }
            for i in range(self.n_features_)
        ]
        rows.sort(key=lambda r: abs(r["mean"]), reverse=True)
        return rows
