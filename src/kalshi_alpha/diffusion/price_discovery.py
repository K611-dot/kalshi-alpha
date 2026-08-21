"""Where price discovery happens: Hasbrouck and Gonzalo-Granger.

When two venues quote the same event, their prices are cointegrated -- they can
wander together but not apart, because the spread between them is bounded by
arbitrage. That structure is what makes "who moves first?" a well-posed
statistical question rather than a chart-reading exercise.

The vector error-correction representation, with the cointegrating vector known
to be :math:`(1, -1)` because both series settle on the same event:

.. math::

    \\Delta p_t = \\alpha (p_{1,t-1} - p_{2,t-1})
                + \\sum_{k=1}^{L} \\Gamma_k \\Delta p_{t-k} + \\varepsilon_t

The adjustment vector :math:`\\alpha` carries the answer. A venue that does
**not** adjust (:math:`\\alpha_i \\approx 0`) is the one setting the price;
the venue that does all the correcting is following. Formally, the vector
orthogonal to :math:`\\alpha`,

.. math::

    \\psi = \\frac{(\\alpha_2, -\\alpha_1)}{\\alpha_2 - \\alpha_1},

gives the permanent shock's loading on each venue.

Two shares are computed from it and they answer different questions:

* **Gonzalo-Granger component share** :math:`\\psi_i` -- how much of the
  *permanent* component each venue contributes. Depends only on error
  correction, not on volatility.
* **Hasbrouck information share** -- the share of the efficient price's
  *innovation variance* attributable to each venue, which also rewards being
  noisy-but-early. It is identified only up to the ordering of the Cholesky
  factorisation when the innovations are correlated, so the honest output is
  the ``[lower, upper]`` bound pair, never a single number.

Reporting a point estimate for the information share while hiding the bounds is
the single most common error in this literature; the bounds are wide exactly
when contemporaneous correlation is high, which is exactly when two venues are
fast and the question is most interesting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

MIN_OBS = 30


@dataclass(frozen=True, slots=True)
class VECMFit:
    alpha: np.ndarray  # (2,) error-correction speeds
    gamma: np.ndarray  # (2, 2*lags) short-run dynamics
    omega: np.ndarray  # (2, 2) residual covariance
    residuals: np.ndarray  # (T, 2)
    lags: int
    n: int
    alpha_stderr: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def alpha_t(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.alpha_stderr > 0, self.alpha / self.alpha_stderr, np.nan)

    @property
    def error_correcting(self) -> bool:
        """Both series pull back toward the spread: the sign pattern we require."""
        return bool(self.alpha[0] < 0 or self.alpha[1] > 0)

    @property
    def psi(self) -> np.ndarray:
        """Common-factor weights, orthogonal to alpha and summing to one."""
        a1, a2 = float(self.alpha[0]), float(self.alpha[1])
        denom = a2 - a1
        if abs(denom) < 1e-12:
            return np.array([0.5, 0.5])
        return np.array([a2 / denom, -a1 / denom])


def _lagged_design(dp: np.ndarray, z: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack the VECM regressors: error-correction term, lagged diffs, constant."""
    t_total = dp.shape[0]
    start = lags
    rows = t_total - start
    cols = 1 + 2 * lags + 1
    X = np.empty((rows, cols))
    X[:, 0] = z[start - 1 : t_total - 1]
    for k in range(1, lags + 1):
        X[:, 1 + 2 * (k - 1)] = dp[start - k : t_total - k, 0]
        X[:, 2 + 2 * (k - 1)] = dp[start - k : t_total - k, 1]
    X[:, -1] = 1.0
    Y = dp[start:]
    return X, Y


def fit_vecm(p1: np.ndarray, p2: np.ndarray, lags: int = 5) -> VECMFit | None:
    """Fit the bivariate VECM with cointegrating vector fixed at ``(1, -1)``.

    Both equations share the same regressors, so seemingly-unrelated regression
    collapses to equation-by-equation OLS and the joint estimate is exact.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if p1.shape != p2.shape:
        raise ValueError("price series must be the same length")
    mask = np.isfinite(p1) & np.isfinite(p2)
    p1, p2 = p1[mask], p2[mask]
    if p1.size < MIN_OBS + lags:
        return None

    z = p1 - p2
    dp = np.column_stack([np.diff(p1), np.diff(p2)])
    z_lag = z[:-1]
    if dp.shape[0] <= lags + 2:
        return None

    X, Y = _lagged_design(dp, z_lag, lags)
    if X.shape[0] < X.shape[1] + 2:
        return None

    xtx_inv = np.linalg.pinv(X.T @ X)
    B = xtx_inv @ X.T @ Y  # (cols, 2)
    resid = Y - X @ B
    dof = max(X.shape[0] - X.shape[1], 1)
    omega = (resid.T @ resid) / dof

    alpha = B[0, :].copy()
    # Standard errors of the two error-correction coefficients.
    se = np.array([np.sqrt(max(omega[j, j] * xtx_inv[0, 0], 0.0)) for j in range(2)])
    gamma = B[1 : 1 + 2 * lags, :].T

    return VECMFit(
        alpha=alpha,
        gamma=gamma,
        omega=omega,
        residuals=resid,
        lags=lags,
        n=int(X.shape[0]),
        alpha_stderr=se,
    )


def component_share(fit: VECMFit) -> np.ndarray:
    """Gonzalo-Granger permanent-component shares (sum to one)."""
    return fit.psi


def information_share(fit: VECMFit) -> dict[str, np.ndarray | float]:
    """Hasbrouck information shares with exact Cholesky-ordering bounds.

    For each ordering of the two venues, factor ``Omega = M M'`` with ``M``
    lower triangular and compute

    .. math:: IS_j = \\frac{([\\psi M]_j)^2}{\\psi \\Omega \\psi'} .

    With only two series there are two orderings, so the bounds are exact
    rather than approximated. The midpoint is reported for convenience but the
    bounds are the result.
    """
    psi = fit.psi
    omega = fit.omega
    denom = float(psi @ omega @ psi)
    if denom <= 0 or not np.isfinite(denom):
        return {"lower": np.array([np.nan, np.nan]), "upper": np.array([np.nan, np.nan]),
                "mid": np.array([np.nan, np.nan]), "denominator": float("nan")}

    shares: list[np.ndarray] = []
    for order in permutations(range(2)):
        idx = np.array(order)
        om = omega[np.ix_(idx, idx)]
        try:
            m = np.linalg.cholesky(om)
        except np.linalg.LinAlgError:
            # Ridge the diagonal just enough to factor a near-singular Omega.
            m = np.linalg.cholesky(om + np.eye(2) * 1e-12 * float(np.trace(om)))
        contrib = (psi[idx] @ m) ** 2 / denom
        back = np.empty(2)
        back[idx] = contrib
        shares.append(back)

    stacked = np.vstack(shares)
    lower = stacked.min(axis=0)
    upper = stacked.max(axis=0)
    return {
        "lower": lower,
        "upper": upper,
        "mid": (lower + upper) / 2.0,
        "denominator": denom,
    }


def adf_test(series: np.ndarray, max_lag: int = 5) -> dict[str, float]:
    """Augmented Dickey-Fuller test with a constant, used on the spread.

    The cointegrating vector is imposed rather than estimated, so this is the
    Engle-Granger second step with *known* first step and the standard ADF
    critical values apply directly -- no Phillips-Ouliaris adjustment needed.
    Rejecting a unit root in the spread is the licence to interpret the VECM.
    """
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < MIN_OBS:
        return {"stat": np.nan, "crit_5pct": -2.86, "stationary": float("nan"), "lags": 0}

    dy = np.diff(y)
    lag = min(max_lag, max(1, int((y.size / 100.0) ** 0.25 * 4)))
    rows = dy.size - lag
    if rows < 10:
        return {"stat": np.nan, "crit_5pct": -2.86, "stationary": float("nan"), "lags": lag}

    X = np.empty((rows, lag + 2))
    X[:, 0] = y[lag:-1]
    for k in range(1, lag + 1):
        X[:, k] = dy[lag - k : dy.size - k]
    X[:, -1] = 1.0
    Y = dy[lag:]

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ Y
    resid = Y - X @ beta
    dof = max(rows - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    se = float(np.sqrt(max(s2 * xtx_inv[0, 0], 0.0)))
    stat = float(beta[0] / se) if se > 0 else np.nan
    crit = -2.86  # 5% constant-only asymptotic critical value (Fuller, 1976)
    return {
        "stat": stat,
        "crit_5pct": crit,
        "stationary": float(np.isfinite(stat) and stat < crit),
        "lags": float(lag),
    }


@dataclass(frozen=True)
class PriceDiscovery:
    venues: tuple[str, str]
    alpha: np.ndarray
    alpha_t: np.ndarray
    component_share: np.ndarray
    is_lower: np.ndarray
    is_upper: np.ndarray
    is_mid: np.ndarray
    residual_corr: float
    adf: dict[str, float]
    n: int
    lags: int

    @property
    def leader(self) -> str:
        return self.venues[int(np.argmax(self.is_mid))]

    @property
    def bounds_width(self) -> float:
        """How much the Cholesky ordering matters; wide means poorly identified."""
        return float(np.max(self.is_upper - self.is_lower))

    def summary(self) -> str:
        a, b = self.venues
        return (
            f"price discovery over n={self.n} bars (lags={self.lags})\n"
            f"  alpha: {a}={self.alpha[0]:+.4f} (t={self.alpha_t[0]:+.2f}), "
            f"{b}={self.alpha[1]:+.4f} (t={self.alpha_t[1]:+.2f})\n"
            f"  Gonzalo-Granger component share: {a}={self.component_share[0]:.3f}  "
            f"{b}={self.component_share[1]:.3f}\n"
            f"  Hasbrouck information share:     {a}=[{self.is_lower[0]:.3f}, "
            f"{self.is_upper[0]:.3f}]  {b}=[{self.is_lower[1]:.3f}, {self.is_upper[1]:.3f}]\n"
            f"  residual corr={self.residual_corr:+.3f}  bound width={self.bounds_width:.3f}\n"
            f"  ADF on spread: stat={self.adf['stat']:.2f} vs 5% crit "
            f"{self.adf['crit_5pct']:.2f} -> "
            f"{'cointegrated' if self.adf.get('stationary') else 'NOT cointegrated'}\n"
            f"  leader: {self.leader}"
        )


def price_discovery(
    p1: np.ndarray,
    p2: np.ndarray,
    venues: tuple[str, str] = ("venue_a", "venue_b"),
    lags: int = 5,
) -> PriceDiscovery | None:
    """Full price-discovery analysis of two cointegrated price series."""
    fit = fit_vecm(p1, p2, lags)
    if fit is None:
        return None
    shares = information_share(fit)
    om = fit.omega
    denom = float(np.sqrt(om[0, 0] * om[1, 1]))
    corr = float(om[0, 1] / denom) if denom > 0 else np.nan
    spread = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    return PriceDiscovery(
        venues=venues,
        alpha=fit.alpha,
        alpha_t=fit.alpha_t,
        component_share=component_share(fit),
        is_lower=np.asarray(shares["lower"]),
        is_upper=np.asarray(shares["upper"]),
        is_mid=np.asarray(shares["mid"]),
        residual_corr=corr,
        adf=adf_test(spread),
        n=fit.n,
        lags=fit.lags,
    )


def bootstrap_information_share(
    p1: np.ndarray,
    p2: np.ndarray,
    lags: int = 5,
    draws: int = 200,
    block: int = 50,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Moving-block bootstrap confidence band for the information share.

    Blocks rather than i.i.d. resampling because the short-run dynamics are
    serially dependent by construction; resampling single observations would
    destroy exactly the autocorrelation the VECM is estimating and produce
    spuriously tight bands.
    """
    rng = np.random.default_rng(seed)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    n = p1.size
    if n < MIN_OBS + lags or block >= n:
        return {"mid": np.array([np.nan, np.nan]), "ci_low": np.array([np.nan, np.nan]),
                "ci_high": np.array([np.nan, np.nan])}

    n_blocks = int(np.ceil(n / block))
    out: list[np.ndarray] = []
    for _ in range(draws):
        starts = rng.integers(0, n - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        fit = fit_vecm(p1[idx], p2[idx], lags)
        if fit is None:
            continue
        shares = information_share(fit)
        mid = np.asarray(shares["mid"])
        if np.all(np.isfinite(mid)):
            out.append(mid)

    if not out:
        return {"mid": np.array([np.nan, np.nan]), "ci_low": np.array([np.nan, np.nan]),
                "ci_high": np.array([np.nan, np.nan])}
    arr = np.vstack(out)
    return {
        "mid": arr.mean(axis=0),
        "ci_low": np.percentile(arr, 2.5, axis=0),
        "ci_high": np.percentile(arr, 97.5, axis=0),
        "draws": np.array([len(out)]),
    }
