"""Exchange fee model.

Kalshi charges a **concave, price-dependent** taker fee

.. math::

    F = \\lceil\\, r \\cdot C \\cdot P \\cdot (1 - P) \\,\\rceil_{\\text{cent}}

with :math:`P` the trade price in dollars, :math:`C` the contract count and
:math:`r = 0.07` on most markets. Two consequences drive the whole strategy:

1. **The fee peaks at even money.** At P = 0.50 it is 1.75c per contract; at
   P = 0.05 it is 0.33c. A four-leg arbitrage assembled out of tail-priced legs
   can survive on an edge that would be entirely eaten if the same legs were
   struck near 50c. The scanner therefore ranks opportunities by *post-fee*
   guaranteed profit, never by gross mispricing.
2. **The ceiling is applied per order, not per contract.** Rounding up to the
   next cent is a fixed cost that is amortised over size, so tiny clips are
   disproportionately penalised. ``exact_taker_fee_cents`` models this exactly;
   ``linear_taker_fee_cents`` drops the ceiling so that the arbitrage LP stays
   linear, and the LP solution is then re-validated with the exact function.

All functions take and return **integer cents**.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from kalshi_alpha.types import PAYOUT, Cents, Leg

DEFAULT_TAKER_RATE = 0.07
DEFAULT_MAKER_RATE = 0.0025


def _ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division for non-negative inputs."""
    return -(-numerator // denominator)


def exact_taker_fee_cents(
    price_cents: Cents, contracts: int, rate: float = DEFAULT_TAKER_RATE
) -> Cents:
    """Exact taker fee in cents, with the exchange's round-up-to-the-cent rule.

    Uses integer arithmetic so results are bit-reproducible across platforms.

    >>> exact_taker_fee_cents(50, 1)      # 0.07 * 0.5 * 0.5 = $0.0175 -> 2c
    2
    >>> exact_taker_fee_cents(50, 100)    # $1.75 -> 175c, no rounding needed
    175
    >>> exact_taker_fee_cents(1, 1)       # deep tail: sub-cent, rounds to 1c
    1
    """
    if contracts <= 0:
        return 0
    if not 0 < price_cents < PAYOUT:
        # Settled/locked prices carry no trading fee.
        return 0
    # rate * C * (p/100) * ((100-p)/100) dollars == rate * C * p * (100-p) / 100 cents
    scaled = int(round(rate * 1_000_000))  # rate in millionths, exact for 0.07
    numerator = scaled * contracts * price_cents * (PAYOUT - price_cents)
    return _ceil_div(numerator, 100 * 1_000_000)


def linear_taker_fee_cents(
    price_cents: Cents, contracts: int = 1, rate: float = DEFAULT_TAKER_RATE
) -> float:
    """Taker fee without the ceiling -- linear in ``contracts``.

    This is the version fed to the arbitrage LP, where a ceiling would make the
    programme integer and non-convex. It is a strict *under*-estimate of the
    exact fee (by at most 1 cent per order), so LP solutions are always
    re-checked against :func:`exact_taker_fee_cents` before being reported.
    """
    if contracts <= 0 or not 0 < price_cents < PAYOUT:
        return 0.0
    return rate * contracts * price_cents * (PAYOUT - price_cents) / 100.0


def maker_fee_cents(
    price_cents: Cents, contracts: int, rate: float = DEFAULT_MAKER_RATE
) -> Cents:
    """Maker fee, charged only on the subset of markets that have one.

    Linear in price rather than concave: ``ceil(rate * C * P)``.
    """
    if contracts <= 0 or not 0 < price_cents < PAYOUT:
        return 0
    scaled = int(round(rate * 1_000_000))
    numerator = scaled * contracts * price_cents
    return _ceil_div(numerator, 1_000_000)


def fee_for_leg(
    price_cents: Cents,
    contracts: int,
    liquidity: str = "taker",
    taker_rate: float = DEFAULT_TAKER_RATE,
    maker_rate: float = DEFAULT_MAKER_RATE,
    maker_fees_enabled: bool = False,
) -> Cents:
    """Dispatch to the right schedule for a single order."""
    if liquidity == "maker":
        return maker_fee_cents(price_cents, contracts, maker_rate) if maker_fees_enabled else 0
    return exact_taker_fee_cents(price_cents, contracts, taker_rate)


def total_fees(legs: Iterable[Leg]) -> Cents:
    return sum(leg.fee_cents for leg in legs)


def round_trip_breakeven_cents(price_cents: Cents, rate: float = DEFAULT_TAKER_RATE) -> float:
    """Cents of price move needed to break even on an aggressive round trip.

    Entering and exiting as a taker at roughly the same price costs
    ``2 * r * P * (1-P)`` cents per contract; the position must move at least
    that far to be profitable. This is the hurdle every non-arbitrage strategy
    in :mod:`kalshi_alpha.backtest.strategies` must clear.

    >>> round(round_trip_breakeven_cents(50), 3)
    3.5
    >>> round(round_trip_breakeven_cents(10), 3)
    1.26
    """
    return 2.0 * linear_taker_fee_cents(price_cents, 1, rate)


def fee_adjusted_ask(price_cents: Cents, rate: float = DEFAULT_TAKER_RATE) -> float:
    """Effective cost per contract when lifting an offer at ``price_cents``."""
    return price_cents + linear_taker_fee_cents(price_cents, 1, rate)


def fee_adjusted_bid(price_cents: Cents, rate: float = DEFAULT_TAKER_RATE) -> float:
    """Effective proceeds per contract when hitting a bid at ``price_cents``."""
    return price_cents - linear_taker_fee_cents(price_cents, 1, rate)


def max_fee_price(rate: float = DEFAULT_TAKER_RATE) -> Cents:
    """Price at which the per-contract taker fee is maximised (even money)."""
    del rate
    return PAYOUT // 2


def fee_curve(rate: float = DEFAULT_TAKER_RATE) -> list[tuple[Cents, float]]:
    """The full per-contract fee curve; used by the HTML report."""
    return [(p, linear_taker_fee_cents(p, 1, rate)) for p in range(1, PAYOUT)]


def assert_fee_monotone_concave(rate: float = DEFAULT_TAKER_RATE) -> None:
    """Sanity check invoked by the test-suite: the curve is concave with a peak at 50c."""
    curve = [linear_taker_fee_cents(p, 1, rate) for p in range(1, PAYOUT)]
    peak = max(range(len(curve)), key=lambda i: curve[i]) + 1
    if peak != PAYOUT // 2:
        raise AssertionError(f"fee peak at {peak}c, expected {PAYOUT // 2}c")
    for i in range(1, len(curve) - 1):
        if curve[i] < min(curve[i - 1], curve[i + 1]) - 1e-9:
            raise AssertionError("fee curve is not concave")


def implied_edge_after_fees(
    gross_edge_cents: float, price_cents: Cents, rate: float = DEFAULT_TAKER_RATE
) -> float:
    """Net edge per contract after paying the taker fee once."""
    return gross_edge_cents - linear_taker_fee_cents(price_cents, 1, rate)


def kelly_fraction(p: float, price_cents: Cents, rate: float = DEFAULT_TAKER_RATE) -> float:
    """Fee-aware Kelly fraction for buying YES at ``price_cents`` with belief ``p``.

    Buying one YES contract costs ``c = price + fee`` cents and pays 100 cents
    with probability ``p``. The bet has odds ``b = (100 - c) / c`` to 1, so the
    Kelly stake is ``f* = (p*b - (1-p)) / b`` clipped at zero.
    """
    cost = price_cents + linear_taker_fee_cents(price_cents, 1, rate)
    if cost <= 0 or cost >= PAYOUT:
        return 0.0
    b = (PAYOUT - cost) / cost
    f = (p * b - (1.0 - p)) / b
    return max(0.0, min(1.0, f)) if math.isfinite(f) else 0.0
