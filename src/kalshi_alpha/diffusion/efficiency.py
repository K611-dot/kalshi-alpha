"""Weak-form efficiency tests.

If information is incorporated instantly, price changes are unpredictable from
their own history. These tests measure the departure from that null, and each
one is sensitive to a different failure mode:

* **Variance ratio** (Lo and MacKinlay, 1988). ``VR(q) > 1`` means multi-period
  moves are larger than the sum of single-period moves -- positive
  autocorrelation, i.e. *underreaction* and continued drift. ``VR(q) < 1``
  means mean reversion, the signature of bid-ask bounce or overreaction. The
  heteroskedasticity-robust statistic is the default because volatility on
  event contracts is wildly non-constant: it collapses as the price approaches
  0 or 100, so the homoskedastic version rejects the null essentially
  everywhere and tells you nothing.
* **Runs test**. Distribution-free. Catches sign predictability even when the
  magnitudes are so heavy-tailed that variance-based tests lose power.
* **Ljung-Box**. Joint test on the first ``h`` autocorrelations; the standard
  portmanteau check that no single lag is being cherry-picked.

An efficient market fails to reject all three. A market that rejects the
variance ratio upward at horizons of a few minutes after a scheduled release is
the one worth trading, and that is precisely the pattern the diffusion pipeline
is built to find.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class TestResult:
    name: str
    statistic: float
    pvalue: float
    n: int
    detail: dict[str, float]

    @property
    def rejects(self) -> bool:
        return bool(np.isfinite(self.pvalue) and self.pvalue < 0.05)

    def summary(self) -> str:
        verdict = "REJECT" if self.rejects else "fail to reject"
        return f"{self.name:22s} stat={self.statistic:+8.3f}  p={self.pvalue:6.4f}  {verdict}"


def _clean(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def autocorrelation(x, nlags: int = 20) -> np.ndarray:
    """Sample autocorrelation function, lags 1..nlags."""
    r = _clean(x)
    n = r.size
    if n < nlags + 2:
        return np.full(nlags, np.nan)
    r = r - r.mean()
    denom = float(r @ r)
    if denom <= 0:
        return np.full(nlags, np.nan)
    return np.array([float(r[k:] @ r[:-k]) / denom for k in range(1, nlags + 1)])


def variance_ratio_test(x, q: int = 2, robust: bool = True) -> TestResult:
    """Lo-MacKinlay overlapping variance-ratio test on a return series."""
    r = _clean(x)
    T = r.size
    if 2 * q + 10 > T or q < 2:
        return TestResult(f"variance_ratio(q={q})", np.nan, np.nan, T, {})

    mu = float(r.mean())
    dev = r - mu
    sigma_a = float(dev @ dev) / (T - 1)
    if sigma_a <= 0:
        return TestResult(f"variance_ratio(q={q})", np.nan, np.nan, T, {})

    rq = np.convolve(r, np.ones(q), mode="valid")  # overlapping q-period sums
    m = q * (T - q + 1) * (1.0 - q / T)
    if m <= 0:
        return TestResult(f"variance_ratio(q={q})", np.nan, np.nan, T, {})
    sigma_c = float(np.sum((rq - q * mu) ** 2)) / m
    vr = sigma_c / sigma_a

    if robust:
        d = float(np.sum(dev**2)) ** 2
        theta = 0.0
        for j in range(1, q):
            num = float(np.sum((dev[j:] ** 2) * (dev[:-j] ** 2)))
            delta_j = T * num / d if d > 0 else np.nan
            theta += ((2.0 * (q - j) / q) ** 2) * delta_j
        var = theta
    else:
        var = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q)

    if not np.isfinite(var) or var <= 0:
        return TestResult(f"variance_ratio(q={q})", vr, np.nan, T, {"vr": vr})

    z = np.sqrt(T) * (vr - 1.0) / np.sqrt(var)
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return TestResult(
        f"variance_ratio(q={q})",
        float(z),
        p,
        T,
        {"vr": float(vr), "variance": float(var), "robust": float(robust)},
    )


def variance_ratio_profile(x, qs=(2, 4, 8, 16, 32), robust: bool = True) -> dict[int, TestResult]:
    """Run the test across horizons; the *shape* across ``q`` is the signal.

    Drift that decays has VR rising then falling as ``q`` passes the half-life,
    which localises the horizon at which the market finishes absorbing news
    without assuming any functional form.
    """
    return {q: variance_ratio_test(x, q, robust) for q in qs}


def runs_test(x, threshold: float | None = None) -> TestResult:
    """Wald-Wolfowitz runs test on the signs of a series."""
    r = _clean(x)
    thr = float(np.median(r)) if threshold is None else float(threshold)
    signs = r > thr
    signs = signs[r != thr] if threshold is None else signs
    n = signs.size
    n1 = int(signs.sum())
    n2 = n - n1
    if n < 20 or n1 == 0 or n2 == 0:
        return TestResult("runs", np.nan, np.nan, n, {})

    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    mean = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n**2 * (n - 1.0))
    if var <= 0:
        return TestResult("runs", np.nan, np.nan, n, {})
    z = (runs - mean) / np.sqrt(var)
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return TestResult("runs", float(z), p, n, {"runs": float(runs), "expected": float(mean)})


def ljung_box(x, lags: int = 10) -> TestResult:
    """Portmanteau test that the first ``lags`` autocorrelations are jointly zero."""
    r = _clean(x)
    n = r.size
    if n < lags + 10:
        return TestResult(f"ljung_box(h={lags})", np.nan, np.nan, n, {})
    acf = autocorrelation(r, lags)
    if not np.all(np.isfinite(acf)):
        return TestResult(f"ljung_box(h={lags})", np.nan, np.nan, n, {})
    q = n * (n + 2) * float(np.sum(acf**2 / (n - np.arange(1, lags + 1))))
    p = float(stats.chi2.sf(q, lags))
    return TestResult(f"ljung_box(h={lags})", q, p, n, {"acf1": float(acf[0])})


def efficiency_panel(returns, qs=(2, 4, 8, 16), lb_lags: int = 10) -> list[TestResult]:
    """Every test at once, in the order a referee would want to read them."""
    out = [variance_ratio_test(returns, q) for q in qs]
    out.append(runs_test(returns))
    out.append(ljung_box(returns, lb_lags))
    return out


def drift_direction(returns, q: int = 8) -> str:
    """Human-readable verdict from the variance ratio at one horizon."""
    res = variance_ratio_test(returns, q)
    vr = res.detail.get("vr", np.nan)
    if not np.isfinite(vr):
        return "undetermined"
    if not res.rejects:
        return "random walk (cannot reject)"
    return "underreaction / drift" if vr > 1 else "overreaction / mean reversion"
