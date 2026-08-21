"""Property-based tests.

Example-based tests check the cases you thought of. These check invariants that
must hold for *every* input, and hypothesis actively searches for the input that
breaks them. On a system whose whole job is to assert "this portfolio is
profitable in every state of the world", that is exactly the right shape of
test: the properties below are the no-arbitrage theorems the code claims to
implement, stated directly.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from kalshi_alpha.arbitrage.detectors import detect_crossed_book, detect_dutch_book
from kalshi_alpha.arbitrage.fees import exact_taker_fee_cents, linear_taker_fee_cents
from kalshi_alpha.arbitrage.payoff import exclusive_settle_map, pnl_over_outcomes
from kalshi_alpha.probability.calibration import brier_score, pava
from kalshi_alpha.probability.constraints import project_to_monotone, project_to_simplex
from kalshi_alpha.types import PAYOUT, Action, EventGroup, Fill, OrderBook, Position, Side

PRICE = st.integers(min_value=1, max_value=99)
SIZE = st.integers(min_value=1, max_value=5_000)
QTY = st.integers(min_value=1, max_value=2_000)
PROB = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

SLOW = hypothesis_settings(max_examples=150, deadline=None)


# --------------------------------------------------------------------------
# fees
# --------------------------------------------------------------------------
@given(price=PRICE, qty=QTY)
def test_exact_fee_is_never_below_the_linear_relaxation(price: int, qty: int) -> None:
    assert exact_taker_fee_cents(price, qty) >= linear_taker_fee_cents(price, qty) - 1e-9


@given(price=PRICE, qty=QTY)
def test_ceiling_overhead_is_bounded_by_one_cent(price: int, qty: int) -> None:
    gap = exact_taker_fee_cents(price, qty) - linear_taker_fee_cents(price, qty)
    assert gap < 1.0 + 1e-9


@given(price=PRICE, a=QTY, b=QTY)
def test_fee_is_subadditive_in_size(price: int, a: int, b: int) -> None:
    """Splitting an order can only ever cost more, because of per-order rounding."""
    assert exact_taker_fee_cents(price, a + b) <= exact_taker_fee_cents(
        price, a
    ) + exact_taker_fee_cents(price, b)


@given(price=PRICE, qty=QTY)
def test_fee_is_symmetric_about_even_money(price: int, qty: int) -> None:
    assert exact_taker_fee_cents(price, qty) == exact_taker_fee_cents(PAYOUT - price, qty)


# --------------------------------------------------------------------------
# book invariants
# --------------------------------------------------------------------------
@given(bid=PRICE, spread=st.integers(min_value=1, max_value=20), size=SIZE)
def test_uncrossed_book_offers_no_within_market_arbitrage(
    bid: int, spread: int, size: int
) -> None:
    """The structural theorem: a normal book can never be a Dutch book against itself."""
    ask = bid + spread
    assume(ask <= 99)
    book = OrderBook.from_levels("T", 0.0, [(bid, size)], [(PAYOUT - ask, size)])
    assert not book.is_crossed
    assert detect_crossed_book(book) == []


@given(bid=PRICE, spread=st.integers(min_value=1, max_value=20), size=SIZE)
def test_yes_ask_and_no_bid_always_sum_to_one_dollar(
    bid: int, spread: int, size: int
) -> None:
    ask = bid + spread
    assume(ask <= 99)
    book = OrderBook.from_levels("T", 0.0, [(bid, size)], [(PAYOUT - ask, size)])
    assert book.best_yes_ask + book.best_no_bid == PAYOUT


@given(bid=PRICE, spread=st.integers(min_value=1, max_value=20), size=SIZE, qty=QTY)
def test_walking_the_book_never_beats_the_touch(
    bid: int, spread: int, size: int, qty: int
) -> None:
    ask = bid + spread
    assume(ask <= 99)
    book = OrderBook.from_levels("T", 0.0, [(bid, size)], [(PAYOUT - ask, size)])
    vwap = book.vwap(Side.YES, Action.BUY, qty)
    if vwap is not None:
        assert vwap >= book.best_yes_ask - 1e-9


@given(bid=PRICE, spread=st.integers(min_value=1, max_value=20),
       bid_size=SIZE, ask_size=SIZE)
def test_microprice_lies_between_bid_and_ask(
    bid: int, spread: int, bid_size: int, ask_size: int
) -> None:
    ask = bid + spread
    assume(ask <= 99)
    book = OrderBook.from_levels("T", 0.0, [(bid, bid_size)], [(PAYOUT - ask, ask_size)])
    assert bid <= book.microprice <= ask


# --------------------------------------------------------------------------
# position accounting
# --------------------------------------------------------------------------
@given(price=PRICE, qty=QTY)
def test_a_hedged_pair_pays_the_same_in_both_states(price: int, qty: int) -> None:
    """The core identity the netting representation must preserve."""
    other = PAYOUT - price
    assume(0 < other < PAYOUT)
    p = Position("T")
    p.apply(Fill("T", 0.0, Side.YES, Action.BUY, qty, price))
    p.apply(Fill("T", 0.0, Side.NO, Action.BUY, qty, other))
    assert p.settle(0) == p.settle(1)


@given(price=PRICE, qty=QTY, fee=st.integers(min_value=0, max_value=500))
def test_buy_then_sell_at_the_same_price_loses_only_fees(
    price: int, qty: int, fee: int
) -> None:
    p = Position("T")
    p.apply(Fill("T", 0.0, Side.YES, Action.BUY, qty, price, fee_cents=fee))
    p.apply(Fill("T", 0.0, Side.YES, Action.SELL, qty, price, fee_cents=fee))
    assert p.settle(1) == -2 * fee
    assert p.settle(0) == -2 * fee


@given(price=PRICE, qty=QTY)
def test_settlement_is_bounded_by_the_payout(price: int, qty: int) -> None:
    p = Position("T")
    p.apply(Fill("T", 0.0, Side.YES, Action.BUY, qty, price))
    assert -qty * price <= p.settle(1) <= qty * PAYOUT


# --------------------------------------------------------------------------
# arbitrage soundness
# --------------------------------------------------------------------------
@given(
    prices=st.lists(PRICE, min_size=2, max_size=5),
    size=st.integers(min_value=50, max_value=2000),
)
@SLOW
def test_reported_arbitrage_is_always_profitable_in_every_state(
    prices: list[int], size: int
) -> None:
    """The only property that really matters: no false positives, ever."""
    tickers = tuple(f"M{i}" for i in range(len(prices)))
    books = {}
    for tk, p in zip(tickers, prices, strict=True):
        ask = min(99, p + 1)
        bid = max(1, p - 1)
        assume(bid < ask)
        books[tk] = OrderBook.from_levels("x", 0.0, [(bid, size)], [(PAYOUT - ask, size)])
        books[tk] = OrderBook(tk, 0.0, books[tk].yes_bids, books[tk].no_bids)

    group = EventGroup("E", tickers, exhaustive=True)
    smap = exclusive_settle_map(list(tickers), exhaustive=True)
    for opp in detect_dutch_book(group, books):
        assert pnl_over_outcomes(opp.legs, smap).min() > 0


# --------------------------------------------------------------------------
# probability projections
# --------------------------------------------------------------------------
@given(values=st.lists(st.floats(-2.0, 2.0, allow_nan=False), min_size=1, max_size=12))
def test_simplex_projection_always_yields_a_distribution(values: list[float]) -> None:
    out = project_to_simplex(values)
    assert (out >= -1e-9).all()
    assert abs(out.sum() - 1.0) < 1e-6


@given(values=st.lists(st.floats(-2.0, 2.0, allow_nan=False), min_size=1, max_size=12))
def test_simplex_projection_is_idempotent(values: list[float]) -> None:
    once = project_to_simplex(values)
    assert np.allclose(project_to_simplex(once), once, atol=1e-9)


@given(values=st.lists(PROB, min_size=1, max_size=12))
def test_monotone_projection_always_yields_a_monotone_sequence(
    values: list[float],
) -> None:
    out = project_to_monotone(values, decreasing=True)
    assert np.all(np.diff(out) <= 1e-9)
    assert ((out >= -1e-9) & (out <= 1.0 + 1e-9)).all()


@given(values=st.lists(st.floats(-5.0, 5.0, allow_nan=False), min_size=1, max_size=40))
def test_pava_is_monotone_and_mean_preserving(values: list[float]) -> None:
    arr = np.asarray(values, dtype=float)
    out = pava(arr)
    assert np.all(np.diff(out) >= -1e-9)
    assert abs(out.mean() - arr.mean()) < 1e-6


@given(values=st.lists(PROB, min_size=2, max_size=10))
def test_projection_never_increases_distance_to_a_valid_point(
    values: list[float],
) -> None:
    """Euclidean projection is a contraction toward the feasible set."""
    arr = np.asarray(values, dtype=float)
    projected = project_to_monotone(arr, decreasing=True)
    reference = np.linspace(1.0, 0.0, arr.size)  # a valid decreasing ladder
    assert np.linalg.norm(projected - reference) <= np.linalg.norm(arr - reference) + 1e-6


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
@given(
    p=st.lists(PROB, min_size=1, max_size=60),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_brier_score_is_bounded(p: list[float], seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = (rng.random(len(p)) < 0.5).astype(float)
    assert 0.0 <= brier_score(p, y) <= 1.0
