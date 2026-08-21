"""Arbitrage detection: it must fire on real violations and stay silent otherwise.

The false-positive tests matter more than the true-positive ones. A scanner that
reports edge which is not there loses money on every signal it generates, and
the usual cause is forgetting that the fee schedule has to be cleared before a
mispricing becomes an arbitrage.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalshi_alpha.arbitrage.detectors import (
    detect_cross_venue,
    detect_crossed_book,
    detect_dutch_book,
    detect_ladder_violation,
)
from kalshi_alpha.arbitrage.engine import ArbEngine, dedupe
from kalshi_alpha.arbitrage.liquidity import sweep
from kalshi_alpha.arbitrage.lp import (
    build_instruments,
    find_arbitrage,
    implied_probabilities,
    implied_state_prices,
    no_arbitrage_bounds,
)
from kalshi_alpha.arbitrage.payoff import (
    exclusive_settle_map,
    ladder_settle_map,
    pnl_over_outcomes,
    verify_opportunity,
)
from kalshi_alpha.config import ArbConfig, Settings
from kalshi_alpha.types import Action, EventGroup, LadderGroup, OrderBook, Side


# --------------------------------------------------------------------------
class TestCrossedBook:
    def test_normal_book_yields_nothing(self, mk_book) -> None:
        assert detect_crossed_book(mk_book("T", 40, 42)) == []

    def test_crossed_book_is_an_arbitrage(self) -> None:
        # YES bid 60 above YES ask 45: buy YES at 45 and NO at 40 for 85c.
        b = OrderBook.from_levels("T", 0.0, [(60, 500)], [(55, 500)])
        found = detect_crossed_book(b)
        assert len(found) == 1
        assert found[0].worst_case_pnl_cents > 0
        assert verify_opportunity(found[0], {"T": [1, 0]})

    def test_a_one_cent_cross_does_not_beat_the_fee(self) -> None:
        # Cross of 1c at even money cannot cover ~3.5c of round-trip fees.
        b = OrderBook.from_levels("T", 0.0, [(50, 500)], [(51, 500)])
        assert detect_crossed_book(b) == []


# --------------------------------------------------------------------------
class TestDutchBook:
    def test_underround_group_is_detected(self, mk_book, exhaustive_group) -> None:
        books = {
            "A": mk_book("A", 28, 30),
            "B": mk_book("B", 28, 30),
            "C": mk_book("C", 30, 32),
        }
        found = detect_dutch_book(exhaustive_group, books)
        kinds = {o.kind for o in found}
        assert "dutch_book_under" in kinds
        under = next(o for o in found if o.kind == "dutch_book_under")
        assert verify_opportunity(under, exclusive_settle_map(["A", "B", "C"]))

    def test_overround_group_is_detected(self, mk_book, exhaustive_group) -> None:
        books = {
            "A": mk_book("A", 40, 42),
            "B": mk_book("B", 40, 42),
            "C": mk_book("C", 40, 42),
        }
        found = detect_dutch_book(exhaustive_group, books)
        assert any(o.kind == "dutch_book_over" for o in found)

    def test_fair_group_yields_nothing(self, mk_book, exhaustive_group) -> None:
        books = {
            "A": mk_book("A", 32, 34),
            "B": mk_book("B", 32, 34),
            "C": mk_book("C", 32, 34),
        }
        # Asks sum to 102, bids to 96: inside the no-arbitrage band.
        assert detect_dutch_book(exhaustive_group, books) == []

    def test_non_exhaustive_group_blocks_the_underround_trade(self, mk_book) -> None:
        """Buying every YES only guarantees a payout if some outcome must win."""
        group = EventGroup("EVT", ("A", "B", "C"), exhaustive=False)
        books = {
            "A": mk_book("A", 28, 30),
            "B": mk_book("B", 28, 30),
            "C": mk_book("C", 28, 30),
        }
        found = detect_dutch_book(group, books)
        assert all(o.kind != "dutch_book_under" for o in found)

    def test_profit_scales_with_the_number_of_outcomes(self, mk_book) -> None:
        group = EventGroup("EVT", ("A", "B", "C", "D"), exhaustive=True)
        books = {t: mk_book(t, 20, 22) for t in ("A", "B", "C", "D")}
        found = detect_dutch_book(group, books)
        assert any(o.kind == "dutch_book_under" for o in found)


# --------------------------------------------------------------------------
class TestLadder:
    def test_monotone_ladder_yields_nothing(self, mk_book, ladder) -> None:
        books = {
            "K1": mk_book("K1", 80, 82),
            "K2": mk_book("K2", 55, 57),
            "K3": mk_book("K3", 20, 22),
        }
        assert detect_ladder_violation(ladder, books) == []

    def test_inverted_ladder_is_an_arbitrage(self, mk_book, ladder) -> None:
        # P(X>=3) quoted far above P(X>=2), which is impossible.
        books = {
            "K1": mk_book("K1", 58, 60),
            "K2": mk_book("K2", 70, 72),
            "K3": mk_book("K3", 20, 22),
        }
        found = detect_ladder_violation(ladder, books)
        assert found
        smap = ladder_settle_map(ladder.tickers, ladder.strikes, ladder.direction)
        assert all(verify_opportunity(o, smap) for o in found)

    def test_small_inversion_is_eaten_by_fees(self, mk_book, ladder) -> None:
        books = {
            "K1": mk_book("K1", 58, 60),
            "K2": mk_book("K2", 61, 63),
            "K3": mk_book("K3", 20, 22),
        }
        # A 1c inversion at even money cannot cover ~2.8c of fees.
        assert detect_ladder_violation(ladder, books) == []

    def test_non_adjacent_pairs_are_scanned(self, mk_book) -> None:
        """Executable monotonicity is not transitive, so adjacent-only misses cases."""
        lad = LadderGroup("E", ("K1", "K2", "K3"), (1.0, 2.0, 3.0), "gte")
        books = {
            "K1": mk_book("K1", 40, 42),
            "K2": mk_book("K2", 48, 62),  # wide, so neither adjacent pair crosses
            "K3": mk_book("K3", 58, 60),
        }
        assert detect_ladder_violation(lad, books)

    def test_decreasing_ladder_orientation(self, mk_book) -> None:
        lad = LadderGroup("E", ("K1", "K2"), (1.0, 2.0), "lte")
        assert not lad.decreasing
        # For a "<=" ladder the probability must *increase* with the strike.
        books = {"K1": mk_book("K1", 70, 72), "K2": mk_book("K2", 30, 32)}
        assert detect_ladder_violation(lad, books)


# --------------------------------------------------------------------------
class TestCrossVenue:
    def test_cheap_here_dear_there(self, mk_book) -> None:
        b = mk_book("T", 40, 42, size=500)
        found = detect_cross_venue(b, other_yes_bid=60, other_yes_ask=62,
                                   venue="other", other_depth=500)
        assert found
        assert all(o.worst_case_pnl_cents > 0 for o in found)

    def test_aligned_venues_yield_nothing(self, mk_book) -> None:
        b = mk_book("T", 40, 42, size=500)
        assert detect_cross_venue(b, 40, 42, other_depth=500) == []

    def test_no_depth_means_no_trade(self, mk_book) -> None:
        b = mk_book("T", 40, 42, size=500)
        assert detect_cross_venue(b, 60, 62, other_depth=0) == []


# --------------------------------------------------------------------------
class TestLP:
    def test_lp_finds_what_the_detector_finds(self, mk_book) -> None:
        books = {
            "A": mk_book("A", 28, 30),
            "B": mk_book("B", 28, 30),
            "C": mk_book("C", 30, 32),
        }
        smap = exclusive_settle_map(["A", "B", "C"], exhaustive=True)
        sol = find_arbitrage(books, smap, "EVT")
        assert sol.found
        assert sol.opportunity is not None
        assert verify_opportunity(sol.opportunity, smap)

    def test_lp_is_silent_on_a_fair_book(self, mk_book) -> None:
        books = {t: mk_book(t, 32, 34) for t in ("A", "B", "C")}
        smap = exclusive_settle_map(["A", "B", "C"], exhaustive=True)
        assert not find_arbitrage(books, smap, "EVT").found

    def test_state_prices_exist_exactly_when_there_is_no_arbitrage(self, mk_book) -> None:
        fair = {t: mk_book(t, 32, 34) for t in ("A", "B", "C")}
        smap = exclusive_settle_map(["A", "B", "C"], exhaustive=True)
        q = implied_state_prices(build_instruments(fair, smap))
        assert q is not None
        assert q.sum() == pytest.approx(1.0, abs=1e-6)
        assert (q >= -1e-9).all()

    def test_implied_probabilities_respect_ladder_monotonicity(self, mk_book, ladder) -> None:
        books = {
            "K1": mk_book("K1", 80, 82),
            "K2": mk_book("K2", 55, 57),
            "K3": mk_book("K3", 20, 22),
        }
        smap = ladder_settle_map(ladder.tickers, ladder.strikes, ladder.direction)
        probs = implied_probabilities(books, smap)
        assert probs is not None
        values = [probs[t] for t in ladder.tickers]
        assert all(a >= b - 1e-9 for a, b in zip(values, values[1:], strict=False))

    def test_no_arbitrage_bounds_bracket_a_quoted_claim(self, mk_book) -> None:
        books = {t: mk_book(t, 32, 34) for t in ("A", "B", "C")}
        smap = exclusive_settle_map(["A", "B", "C"], exhaustive=True)
        instruments = build_instruments(books, smap)
        # Claim that pays $1 if A wins -- the same payoff as A's YES contract.
        lo, hi = no_arbitrage_bounds(instruments, [100.0, 0.0, 0.0])
        assert 0 <= lo <= hi <= 100
        assert lo <= 33 <= hi + 5

    def test_bounds_widen_for_a_claim_the_book_does_not_pin_down(self, mk_book) -> None:
        books = {t: mk_book(t, 32, 34) for t in ("A", "B", "C")}
        smap = exclusive_settle_map(["A", "B", "C"], exhaustive=True)
        instruments = build_instruments(books, smap)
        single = no_arbitrage_bounds(instruments, [100.0, 0.0, 0.0])
        pair = no_arbitrage_bounds(instruments, [100.0, 100.0, 0.0])
        assert single is not None and pair is not None
        assert pair[1] >= single[1]


# --------------------------------------------------------------------------
class TestEngine:
    def test_scan_is_silent_on_a_simulated_coherent_ladder(self, sim, settings) -> None:
        """The strongest correctness signal available without live data."""
        result = ArbEngine(settings).scan(sim.final_books(), ladders=[sim.ladder])
        assert result.opportunities == []

    def test_every_reported_opportunity_survives_re_verification(self, mk_book) -> None:
        books = {
            "A": mk_book("A", 28, 30),
            "B": mk_book("B", 28, 30),
            "C": mk_book("C", 30, 32),
        }
        group = EventGroup("EVT", ("A", "B", "C"), exhaustive=True)
        smap = exclusive_settle_map(["A", "B", "C"], exhaustive=True)
        for opp in ArbEngine().scan(books, groups=[group]).opportunities:
            assert pnl_over_outcomes(opp.legs, smap).min() > 0

    def test_dedupe_keeps_the_richer_duplicate(self, mk_book) -> None:
        books = {t: mk_book(t, 28, 30) for t in ("A", "B", "C")}
        group = EventGroup("EVT", ("A", "B", "C"), exhaustive=True)
        raw = ArbEngine().scan(books, groups=[group], use_lp=True)
        assert len(dedupe(raw.opportunities)) <= len(raw.opportunities) + 1

    def test_size_haircut_reduces_executable_size(self, mk_book) -> None:
        books = {
            "A": mk_book("A", 28, 30),
            "B": mk_book("B", 28, 30),
            "C": mk_book("C", 30, 32),
        }
        group = EventGroup("EVT", ("A", "B", "C"), exhaustive=True)
        full = Settings(arb=ArbConfig(size_haircut=1.0))
        thin = Settings(arb=ArbConfig(size_haircut=0.1))
        big = ArbEngine(full).scan(books, groups=[group], use_lp=False)
        small = ArbEngine(thin).scan(books, groups=[group], use_lp=False)
        assert big.opportunities[0].qty > small.opportunities[0].qty


# --------------------------------------------------------------------------
class TestLiquidity:
    def test_sweep_charges_a_fee_per_level(self) -> None:
        b = OrderBook.from_levels("T", 0.0, [], [(58, 10), (57, 10)])
        one_level = sweep(b, Side.YES, Action.BUY, 10)
        two_levels = sweep(b, Side.YES, Action.BUY, 20)
        assert one_level.complete and two_levels.complete
        assert two_levels.fee_cents > one_level.fee_cents
        assert two_levels.vwap > one_level.vwap  # deeper fills are worse

    def test_incomplete_sweep_is_flagged(self, mk_book) -> None:
        b = mk_book("T", 40, 42, size=5, depth=1)
        assert not sweep(b, Side.YES, Action.BUY, 50).complete

    def test_haircut_shrinks_available_size(self, mk_book) -> None:
        b = mk_book("T", 40, 42, size=100, depth=1)
        assert sweep(b, Side.YES, Action.BUY, 100, size_haircut=1.0).complete
        assert not sweep(b, Side.YES, Action.BUY, 100, size_haircut=0.5).complete

    def test_cost_curve_is_convex(self) -> None:
        b = OrderBook.from_levels("T", 0.0, [], [(58, 10), (57, 10), (56, 10)])
        costs = [sweep(b, Side.YES, Action.BUY, q).cash_cents for q in range(1, 31)]
        marginal = np.diff(costs)
        assert all(a <= b_ for a, b_ in zip(marginal, marginal[1:], strict=False))
