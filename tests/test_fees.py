"""Fee schedule.

These are exact-value tests, not tolerance tests. The fee formula is the hurdle
every threshold in the system is derived from, so an off-by-one-cent error here
propagates into every arbitrage decision the scanner makes.
"""

from __future__ import annotations

import math

import pytest

from kalshi_alpha.arbitrage.fees import (
    assert_fee_monotone_concave,
    exact_taker_fee_cents,
    fee_adjusted_ask,
    fee_adjusted_bid,
    kelly_fraction,
    linear_taker_fee_cents,
    maker_fee_cents,
    round_trip_breakeven_cents,
)


@pytest.mark.parametrize(
    ("price", "contracts", "expected"),
    [
        (50, 1, 2),  # 0.07 * 0.5 * 0.5 = $0.0175 -> rounds up to 2c
        (50, 100, 175),  # $1.75 exactly, no rounding
        (50, 200, 350),
        (1, 1, 1),  # deep tail is sub-cent but still charged a full cent
        (99, 1, 1),
        (25, 100, 132),  # 0.07 * 100 * 0.25 * 0.75 = $1.3125 -> 132c
        (10, 1000, 630),  # 0.07 * 1000 * 0.1 * 0.9 = $6.30 exactly
    ],
)
def test_exact_taker_fee(price: int, contracts: int, expected: int) -> None:
    assert exact_taker_fee_cents(price, contracts) == expected


def test_fee_is_zero_outside_tradeable_range() -> None:
    assert exact_taker_fee_cents(0, 100) == 0
    assert exact_taker_fee_cents(100, 100) == 0
    assert exact_taker_fee_cents(50, 0) == 0
    assert exact_taker_fee_cents(50, -5) == 0


def test_fee_is_symmetric_about_even_money() -> None:
    for p in range(1, 50):
        assert exact_taker_fee_cents(p, 500) == exact_taker_fee_cents(100 - p, 500)


def test_fee_peaks_at_fifty_and_is_concave() -> None:
    assert_fee_monotone_concave()


def test_linear_fee_never_exceeds_exact() -> None:
    """The LP relaxation must be optimistic, never pessimistic."""
    for price in range(1, 100, 7):
        for contracts in (1, 5, 50, 500):
            assert linear_taker_fee_cents(price, contracts) <= exact_taker_fee_cents(
                price, contracts
            )


def test_ceiling_costs_at_most_one_cent_per_order() -> None:
    for price in range(1, 100, 3):
        for contracts in (1, 7, 100, 999):
            gap = exact_taker_fee_cents(price, contracts) - linear_taker_fee_cents(
                price, contracts
            )
            # Tolerance on the lower bound only: when the linear fee lands on an
            # exact cent the two agree, up to float representation error.
            assert -1e-9 <= gap < 1.0 + 1e-9


def test_fee_per_contract_falls_with_size() -> None:
    """The rounding overhead amortises, so bigger clips are cheaper per contract."""
    small = exact_taker_fee_cents(37, 1) / 1
    large = exact_taker_fee_cents(37, 1000) / 1000
    assert large < small


def test_round_trip_breakeven() -> None:
    assert round_trip_breakeven_cents(50) == pytest.approx(3.5)
    assert round_trip_breakeven_cents(10) == pytest.approx(1.26)
    assert round_trip_breakeven_cents(90) == pytest.approx(1.26)
    # Tails are dramatically cheaper to trade than even money.
    assert round_trip_breakeven_cents(5) < round_trip_breakeven_cents(50) / 2


def test_fee_adjusted_prices_bracket_the_raw_price() -> None:
    for p in range(5, 96, 10):
        assert fee_adjusted_bid(p) < p < fee_adjusted_ask(p)


def test_maker_fee_is_linear_in_price() -> None:
    assert maker_fee_cents(50, 10_000) == math.ceil(0.0025 * 10_000 * 50)
    assert maker_fee_cents(20, 10_000) < maker_fee_cents(80, 10_000)


class TestKelly:
    def test_no_edge_means_no_bet(self) -> None:
        # Believing exactly the offered price leaves nothing after fees.
        assert kelly_fraction(0.50, 50) == 0.0

    def test_edge_produces_a_positive_stake(self) -> None:
        assert kelly_fraction(0.70, 50) > 0.0

    def test_stake_grows_with_conviction(self) -> None:
        assert kelly_fraction(0.90, 50) > kelly_fraction(0.60, 50)

    def test_never_exceeds_full_bankroll(self) -> None:
        assert kelly_fraction(1.0, 1) <= 1.0

    def test_fees_shrink_the_stake(self) -> None:
        with_fee = kelly_fraction(0.70, 50, rate=0.07)
        without = kelly_fraction(0.70, 50, rate=0.0)
        assert with_fee < without
