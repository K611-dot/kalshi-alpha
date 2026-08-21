"""Simulator with known ground truth.

Every estimator in this package makes a claim about an unobservable quantity:
*this* venue leads, information is absorbed with *that* half-life, the market is
mispriced by *this* much. On live data none of those can be checked -- the truth
is not observable, so a subtly wrong estimator looks exactly like a correct one.

So the package ships a generator in which the truth is known by construction:

* a latent underlying follows a driftless GBM, and each market's fair
  probability is the closed-form digital price :math:`\\Phi(d_2)` off that
  underlying. The whole strike ladder is therefore internally coherent and
  monotone **by construction** -- the arbitrage scanner should find nothing,
  and when :func:`inject_dislocation` deliberately breaks one strike it should
  find exactly that one;
* a scheduled release jumps the underlying and the quoted price converges to
  its new fair value with a **specified** half-life, so the diffusion
  estimators can be scored against the number that generated the data;
* two venues each receive their own information shocks and error-correct toward
  the efficient price, so the Hasbrouck and Gonzalo-Granger shares have a known
  target.

The tests in ``tests/test_recovery.py`` assert recovery within tolerance. That
turns "the code runs" into "the estimator is correct", which is the only claim
worth making.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from kalshi_alpha.types import (
    MAX_PRICE,
    MIN_PRICE,
    PAYOUT,
    EventGroup,
    LadderGroup,
    MarketMeta,
    OrderBook,
    Side,
    Trade,
    clamp_price,
)


# --------------------------------------------------------------------------
# fair value
# --------------------------------------------------------------------------
def digital_probability(spot: float, strike: float, sigma: float, tau: float) -> float:
    """Risk-neutral ``P(S_T >= K)`` for a driftless GBM.

    ``Phi(d2)`` with ``d2 = (ln(S/K) - sigma^2 tau / 2) / (sigma sqrt(tau))``.
    Driftless because a prediction-market price is a martingale under the
    measure the market itself defines -- there is no risk premium to add.
    """
    if tau <= 0:
        return float(spot >= strike)
    if sigma <= 0:
        return float(spot >= strike)
    if spot <= 0 or strike <= 0:
        return float(spot >= strike)
    d2 = (np.log(spot / strike) - 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    return float(norm.cdf(d2))


@dataclass
class SimConfig:
    """Everything that defines a simulated session."""

    n_steps: int = 2_400
    dt_s: float = 5.0
    start_ts: float = 1_760_000_000.0

    # underlying
    spot: float = 100.0
    sigma: float = 0.35  # per sqrt(year)
    horizon_years: float = 0.02  # ~1 trading week to settlement
    # Time to settlement decays to this fraction of the horizon over the
    # session. As tau shrinks, a digital's sensitivity to the underlying rises,
    # so the price series becomes progressively more volatile -- a real effect,
    # and the reason single-series half-life estimates degrade over long
    # windows. Raise this toward 1.0 to hold the regime steady when isolating
    # an estimator from it.
    tau_floor_frac: float = 0.05

    # market microstructure
    spread_cents: int = 2
    depth_levels: int = 4
    base_size: int = 400
    quote_noise_cents: float = 0.6
    trade_rate: float = 0.15  # expected trades per step
    trade_size_mean: float = 25.0

    # scheduled release
    event_step: int | None = 900
    # Size of the release, expressed directly as the number of cents the
    # at-the-money contract should reprice by. Specifying the shock in
    # probability space rather than in units of underlying volatility keeps it
    # interpretable: a 2c release is noise, an 18c release is a CPI surprise.
    event_jump_cents: float = 18.0
    adjustment_half_life_s: float = 120.0

    seed: int = 7

    @property
    def total_seconds(self) -> float:
        return self.n_steps * self.dt_s


@dataclass
class SimulatedLadder:
    """Output of :func:`simulate_ladder`."""

    times: np.ndarray
    underlying: np.ndarray
    fair: dict[str, np.ndarray]  # ticker -> true probability path
    quoted: dict[str, np.ndarray]  # ticker -> quoted probability path (with lag + noise)
    books: list[dict[str, OrderBook]]
    trades: dict[str, list[Trade]]
    meta: dict[str, MarketMeta]
    ladder: LadderGroup
    group: EventGroup
    event_ts: list[float] = field(default_factory=list)
    truth: dict[str, float] = field(default_factory=dict)

    @property
    def tickers(self) -> tuple[str, ...]:
        return self.ladder.tickers

    def books_at(self, index: int) -> dict[str, OrderBook]:
        return self.books[index]

    def final_books(self) -> dict[str, OrderBook]:
        return self.books[-1]

    def price_path(self, ticker: str) -> np.ndarray:
        """Quoted mid in cents."""
        return self.quoted[ticker] * PAYOUT

    def settle(self) -> dict[str, int]:
        """Realised YES outcome per market, from the terminal underlying."""
        final = float(self.underlying[-1])
        return {tk: self.meta[tk].payoff(final) for tk in self.tickers}


def _make_book(
    ticker: str,
    ts: float,
    prob: float,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> OrderBook:
    """Build a plausible two-sided book around a fair probability."""
    mid = float(np.clip(prob * PAYOUT + rng.normal(0.0, cfg.quote_noise_cents), 2.0, 98.0))
    half = max(1, cfg.spread_cents) / 2.0
    bid = clamp_price(mid - half)
    ask = clamp_price(mid + half)
    if ask <= bid:
        ask = min(MAX_PRICE, bid + 1)
    if bid >= ask:
        bid = max(MIN_PRICE, ask - 1)

    yes_bids: list[tuple[int, int]] = []
    no_bids: list[tuple[int, int]] = []
    for k in range(cfg.depth_levels):
        # Depth thickens away from the touch, as it does on a real book.
        size = int(max(1, rng.poisson(cfg.base_size * (1.0 + 0.6 * k))))
        yb = bid - k
        if MIN_PRICE <= yb <= MAX_PRICE:
            yes_bids.append((yb, size))
        size2 = int(max(1, rng.poisson(cfg.base_size * (1.0 + 0.6 * k))))
        nb = (PAYOUT - ask) - k
        if MIN_PRICE <= nb <= MAX_PRICE:
            no_bids.append((nb, size2))
    return OrderBook.from_levels(ticker, ts, yes_bids, no_bids)


def _lagged_response(
    fair: np.ndarray, half_life_s: float, dt_s: float, event_index: int | None
) -> np.ndarray:
    """Quoted probability that tracks fair value with exponential adjustment.

    Before the release the quote is efficient. From the release onward it
    follows the partial-adjustment recursion

    .. math:: q_t = \\phi q_{t-1} + (1 - \\phi) f_t ,
        \\qquad \\phi = 2^{-\\Delta t / H}

    so the quote absorbs only a fraction :math:`1 - \\phi` of the jump
    immediately and the residual gap :math:`q_t - f_t` then decays
    geometrically at rate :math:`\\phi`. That gap is an exact AR(1) with
    half-life ``H``, which is precisely the object
    :func:`~kalshi_alpha.diffusion.halflife.ar1_half_life` inverts -- so the
    estimator has a known right answer to be scored against.
    """
    out = fair.copy()
    if event_index is None or half_life_s <= 0 or event_index >= fair.size:
        return out
    phi = 0.5 ** (dt_s / half_life_s)
    for i in range(max(event_index, 1), fair.size):
        out[i] = phi * out[i - 1] + (1.0 - phi) * fair[i]
    return out


def simulate_ladder(
    strikes: Sequence[float] = (96.0, 98.0, 100.0, 102.0, 104.0),
    cfg: SimConfig | None = None,
    event_ticker: str = "SIMIDX",
) -> SimulatedLadder:
    """Simulate one strike ladder over a session containing one release."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)

    n = cfg.n_steps
    times = cfg.start_ts + np.arange(n) * cfg.dt_s
    step_vol = cfg.sigma * np.sqrt(cfg.dt_s / (365.25 * 24 * 3600))

    shocks = rng.normal(0.0, step_vol, size=n)

    # Time to settlement shrinks as the session progresses.
    tau = np.linspace(cfg.horizon_years, cfg.horizon_years * cfg.tau_floor_frac, n)

    if cfg.event_step is not None and 0 < cfg.event_step < n and cfg.event_jump_cents:
        # A scheduled release is a jump, not a sequence of small moves. Size it
        # by inverting the digital's delta at the central strike:
        # dP/dlnS = phi(d2) / (sigma sqrt(tau)), so the log-move that moves the
        # contract by the requested number of cents is the target divided by it.
        atm_strike = float(np.median(np.asarray(strikes, dtype=float)))
        t_evt = float(tau[cfg.event_step])
        vol_t = cfg.sigma * np.sqrt(max(t_evt, 1e-12))
        spot_evt = cfg.spot * float(np.exp(np.sum(shocks[: cfg.event_step])))
        d2 = (np.log(spot_evt / atm_strike) - 0.5 * cfg.sigma**2 * t_evt) / vol_t
        sensitivity = float(norm.pdf(d2) / vol_t)  # probability per unit log-move
        if sensitivity > 1e-9:
            target = (cfg.event_jump_cents / PAYOUT) / sensitivity
            shocks[cfg.event_step] += float(rng.choice([-1.0, 1.0])) * target

    log_path = np.cumsum(shocks)
    underlying = cfg.spot * np.exp(log_path)

    tickers = tuple(f"{event_ticker}-{int(k * 100):05d}" for k in strikes)
    fair: dict[str, np.ndarray] = {}
    quoted: dict[str, np.ndarray] = {}
    meta: dict[str, MarketMeta] = {}

    for tk, k in zip(tickers, strikes, strict=True):
        path = np.array(
            [digital_probability(float(s), float(k), cfg.sigma, float(t))
             for s, t in zip(underlying, tau, strict=True)]
        )
        fair[tk] = path
        quoted[tk] = np.clip(
            _lagged_response(path, cfg.adjustment_half_life_s, cfg.dt_s, cfg.event_step),
            0.01,
            0.99,
        )
        meta[tk] = MarketMeta(
            ticker=tk,
            event_ticker=event_ticker,
            title=f"Index >= {k}",
            strike=float(k),
            strike_type="gte",
            close_ts=float(times[-1]),
            category="simulated",
        )

    books: list[dict[str, OrderBook]] = []
    trades: dict[str, list[Trade]] = {tk: [] for tk in tickers}
    for i in range(n):
        snapshot = {tk: _make_book(tk, float(times[i]), float(quoted[tk][i]), cfg, rng)
                    for tk in tickers}
        books.append(snapshot)
        for tk in tickers:
            if rng.random() < cfg.trade_rate:
                book = snapshot[tk]
                # Aggressors lean toward closing the gap between quote and fair.
                gap = fair[tk][i] - quoted[tk][i]
                p_buy = float(np.clip(0.5 + 4.0 * gap, 0.05, 0.95))
                taker = Side.YES if rng.random() < p_buy else Side.NO
                px = book.best_yes_ask if taker is Side.YES else book.best_yes_bid
                if px is None:
                    continue
                trades[tk].append(
                    Trade(
                        ticker=tk,
                        ts=float(times[i]),
                        price=int(px),
                        size=int(max(1, rng.poisson(cfg.trade_size_mean))),
                        taker_side=taker,
                    )
                )

    ladder = LadderGroup(
        event_ticker=event_ticker,
        tickers=tickers,
        strikes=tuple(float(k) for k in strikes),
        direction="gte",
        label="simulated index ladder",
    )
    # The buckets between adjacent strikes form an exhaustive partition.
    group = EventGroup(event_ticker=event_ticker, tickers=tickers, exhaustive=False,
                       label="ladder (not exhaustive as levels, only as buckets)")

    event_ts = [float(times[cfg.event_step])] if cfg.event_step is not None else []
    return SimulatedLadder(
        times=times,
        underlying=underlying,
        fair=fair,
        quoted=quoted,
        books=books,
        trades=trades,
        meta=meta,
        ladder=ladder,
        group=group,
        event_ts=event_ts,
        truth={
            "adjustment_half_life_s": cfg.adjustment_half_life_s,
            "event_ts": event_ts[0] if event_ts else np.nan,
            "spread_cents": float(cfg.spread_cents),
        },
    )


def inject_ladder_violation(
    books: dict[str, OrderBook],
    tickers: Sequence[str],
    magnitude_cents: int,
) -> tuple[dict[str, OrderBook], str | None]:
    """Break ladder monotonicity by a **specified** number of cents.

    Shifting a strike by a fixed amount does not reliably create a violation:
    if the strike sits far below its neighbour, a 30-cent push just moves it to
    a still-monotone level, and the scanner correctly reports nothing. That
    makes it useless as a test instrument.

    Instead this computes the shift required to lift one strike's *bid* to
    ``magnitude_cents`` above the next-lower strike's *ask*, which is exactly
    the executable condition the vertical-spread detector tests. The magnitude
    is then directly comparable against the fee hurdle.

    Returns the modified books and the ticker that was moved (``None`` when no
    suitable pair exists).
    """
    usable = [
        t for t in tickers
        if t in books and books[t].is_two_sided and books[t].best_yes_bid is not None
    ]
    if len(usable) < 2 or magnitude_cents <= 0:
        return dict(books), None

    # Prefer the pair whose required shift keeps both quotes on the price grid.
    best: tuple[int, str, int] | None = None
    for lower, higher in zip(usable, usable[1:], strict=False):
        lower_ask = books[lower].best_yes_ask
        higher_bid = books[higher].best_yes_bid
        if lower_ask is None or higher_bid is None:
            continue
        shift = (lower_ask + magnitude_cents) - higher_bid
        if shift <= 0:
            continue
        target_bid = higher_bid + shift
        if target_bid >= MAX_PRICE:
            continue
        if best is None or shift < best[0]:
            best = (shift, higher, target_bid)

    if best is None:
        return dict(books), None
    shift, ticker, _ = best
    return inject_dislocation(books, ticker, shift), ticker


def inject_dislocation(
    books: dict[str, OrderBook],
    ticker: str,
    shift_cents: int,
) -> dict[str, OrderBook]:
    """Move one market's quotes by ``shift_cents`` to create a real violation.

    Used to prove the detectors fire on a known dislocation rather than merely
    failing to fire on clean data. Shifting a *single* strike is the cleanest
    test: it breaks ladder monotonicity without touching any other invariant.
    """
    out = dict(books)
    book = out[ticker]
    yb = [(min(MAX_PRICE, max(MIN_PRICE, lv.price + shift_cents)), lv.size)
          for lv in book.yes_bids]
    nb = [(min(MAX_PRICE, max(MIN_PRICE, lv.price - shift_cents)), lv.size)
          for lv in book.no_bids]
    out[ticker] = OrderBook.from_levels(ticker, book.ts, yb, nb)
    return out


# --------------------------------------------------------------------------
# two-venue price discovery
# --------------------------------------------------------------------------
@dataclass
class TwoVenueSim:
    times: np.ndarray
    efficient: np.ndarray
    venue_a: np.ndarray
    venue_b: np.ndarray
    true_share_a: float
    kappa_a: float
    kappa_b: float

    @property
    def true_share_b(self) -> float:
        return 1.0 - self.true_share_a


def simulate_two_venue(
    n: int = 4_000,
    sigma_a: float = 0.9,
    sigma_b: float = 0.3,
    kappa_a: float = 0.05,
    kappa_b: float = 0.35,
    noise: float = 0.05,
    dt_s: float = 1.0,
    start_ts: float = 1_760_000_000.0,
    seed: int = 11,
) -> TwoVenueSim:
    """Two venues quoting one event, each contributing its own information.

    Venue ``i`` originates innovations of size ``sigma_i`` and error-corrects
    toward the efficient price at rate ``kappa_i``. The efficient price is the
    sum of both innovation streams, so the *true* information share of A is
    ``sigma_a^2 / (sigma_a^2 + sigma_b^2)``, and a venue that corrects slowly
    (small ``kappa``) is the one leading.

    Observation noise is added on top so the estimators face the same
    measurement problem they face live -- without it the VECM residuals would be
    degenerate and the Cholesky bounds would collapse artificially.
    """
    rng = np.random.default_rng(seed)
    ea = rng.normal(0.0, sigma_a, size=n)
    eb = rng.normal(0.0, sigma_b, size=n)
    efficient = np.cumsum(ea + eb)

    a = np.empty(n)
    b = np.empty(n)
    a[0] = b[0] = efficient[0]
    for t in range(1, n):
        a[t] = a[t - 1] + ea[t] + kappa_a * (efficient[t - 1] - a[t - 1])
        b[t] = b[t - 1] + eb[t] + kappa_b * (efficient[t - 1] - b[t - 1])

    # The venues' latent paths are unobservable; what an estimator sees is the
    # path plus quoting noise. Keeping them as separate names makes that
    # distinction explicit rather than shadowing the latent series.
    observed_a = a + rng.normal(0.0, noise, size=n)
    observed_b = b + rng.normal(0.0, noise, size=n)
    times = start_ts + np.arange(n) * dt_s
    share_a = sigma_a**2 / (sigma_a**2 + sigma_b**2)
    return TwoVenueSim(
        times, efficient, observed_a, observed_b, float(share_a), kappa_a, kappa_b
    )


def simulate_calibration_sample(
    n: int = 4_000,
    bias_a: float = 0.85,
    bias_b: float = 0.0,
    seed: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Quoted probabilities and realised outcomes with a known calibration flaw.

    ``bias_a < 1`` produces the classic favourite-longshot pattern: quoted
    log-odds are ``1 / bias_a`` times too extreme, so longshots are overpriced
    and favourites underpriced. Platt scaling fits ``sigma(a logit(q) + b)``
    against the outcomes and should therefore recover ``a ~= bias_a`` -- the
    shrinkage that undoes the exaggeration.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, 1.6, size=n)
    true_p = 1.0 / (1.0 + np.exp(-latent))
    quoted_logit = (latent - bias_b) / max(bias_a, 1e-6)
    quoted = 1.0 / (1.0 + np.exp(-quoted_logit))
    outcomes = (rng.random(n) < true_p).astype(float)
    return quoted, outcomes
