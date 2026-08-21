"""Wire-format parsing, pinned to payloads captured from the live exchange.

These fixtures are **real responses**, copied verbatim from
``api.elections.kalshi.com`` rather than written from the documentation. That
matters: the API moved from integer cents under ``yes``/``no`` to fixed-point
dollar strings under ``orderbook_fp.yes_dollars``/``no_dollars``, and the old
parser did not error on the new shape -- it returned an empty book. A silently
empty book makes the scanner find no arbitrage anywhere, which is
indistinguishable from a working system that has correctly found nothing.

So the parsers are pinned against both encodings, and the invariants are
checked against the market payload's own top-of-book fields, which is the one
cross-check the exchange gives us for free.
"""

from __future__ import annotations

import pytest

from kalshi_alpha.data.kalshi_client import (
    dollars_to_cents,
    parse_market_meta,
    parse_orderbook,
    parse_quote,
    parse_size,
    parse_timestamp,
    parse_trade,
)
from kalshi_alpha.types import PAYOUT

# --- captured live 2026-08-21, KXGDPYEAR-36-B4.8 --------------------------
LIVE_ORDERBOOK = {
    "orderbook_fp": {
        "no_dollars": [
            ["0.0100", "1.00"],
            ["0.1200", "2989.00"],
            ["0.1400", "51.00"],
            ["0.1500", "35.00"],
            ["0.9600", "479.97"],
            ["0.9700", "247.00"],
        ],
        "yes_dollars": [["0.0100", "677.52"]],
    }
}

# The same market's own top-of-book fields, which must agree with the ladder.
LIVE_MARKET = {
    "ticker": "KXGDPYEAR-36-B4.8",
    "event_ticker": "KXGDPYEAR-36",
    "title": "GDP growth in 2036?",
    "strike_type": "between",
    "floor_strike": 4.6,
    "cap_strike": 5,
    "close_time": "2037-12-31T13:29:00Z",
    "yes_bid_dollars": "0.0100",
    "yes_ask_dollars": "0.0300",
    "no_bid_dollars": "0.9700",
    "no_ask_dollars": "0.9900",
}

LIVE_TRADE = {
    "count_fp": "21.00",
    "created_time": "2026-08-18T17:05:57.133601Z",
    "no_price_dollars": "0.9800",
    "taker_side": "yes",
    "ticker": "KXGDPYEAR-36-B4.8",
    "yes_price_dollars": "0.0200",
}

LEGACY_ORDERBOOK = {"yes": [[40, 100], [39, 50]], "no": [[58, 80]]}


class TestUnitConversion:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("0.0100", 1), ("0.9700", 97), ("0.0300", 3), ("0.5000", 50), (0.01, 1), (0.99, 99)],
    )
    def test_dollars_to_cents(self, value, expected) -> None:
        assert dollars_to_cents(value) == expected

    def test_no_truncation_on_binary_representation(self) -> None:
        """0.03 * 100 is 2.9999... in binary; truncating would shift the book."""
        for cents in range(1, 100):
            assert dollars_to_cents(f"{cents / 100:.4f}") == cents

    def test_bad_input_is_none_not_an_exception(self) -> None:
        assert dollars_to_cents(None) is None
        assert dollars_to_cents("") is None
        assert dollars_to_cents("abc") is None

    def test_fractional_size_floors(self) -> None:
        # Understating depth costs opportunity; overstating it costs money.
        assert parse_size("677.52") == 677
        assert parse_size("479.97") == 479
        assert parse_size("1.00") == 1
        assert parse_size("0.40") == 0

    def test_size_never_negative(self) -> None:
        assert parse_size("-5") == 0
        assert parse_size(None) == 0
        assert parse_size("junk") == 0

    def test_timestamp_accepts_iso_and_epoch(self) -> None:
        iso = parse_timestamp({"created_time": "2026-08-18T17:05:57.133601Z"}, "created_time")
        epoch = parse_timestamp({"ts": 1787072757.0}, "ts")
        assert iso == pytest.approx(1787072757.13, abs=1.0)
        assert epoch == pytest.approx(1787072757.0)

    def test_timestamp_prefers_the_first_present_key(self) -> None:
        got = parse_timestamp({"b": 5.0}, "a", "b")
        assert got == pytest.approx(5.0)


class TestLiveOrderbook:
    def test_current_format_parses(self) -> None:
        book = parse_orderbook("KXGDPYEAR-36-B4.8", LIVE_ORDERBOOK, 123.0)
        assert book.best_yes_bid == 1
        assert book.best_no_bid == 97
        assert book.best_yes_ask == 3
        assert book.ts == 123.0

    def test_ladder_agrees_with_the_market_payload(self) -> None:
        """The exchange reports the touch twice; the two must not disagree."""
        book = parse_orderbook("T", LIVE_ORDERBOOK, 0.0)
        bid, ask = parse_quote(LIVE_MARKET)
        assert book.best_yes_bid == bid
        assert book.best_yes_ask == ask

    def test_sizes_are_whole_contracts(self) -> None:
        book = parse_orderbook("T", LIVE_ORDERBOOK, 0.0)
        assert book.best_yes_bid_size == 677
        assert book.best_yes_ask_size == 247

    def test_full_depth_is_kept(self) -> None:
        book = parse_orderbook("T", LIVE_ORDERBOOK, 0.0)
        assert len(book.no_bids) == 6
        assert len(book.yes_bids) == 1

    def test_book_is_not_crossed(self) -> None:
        assert not parse_orderbook("T", LIVE_ORDERBOOK, 0.0).is_crossed

    def test_legacy_integer_cent_format_still_parses(self) -> None:
        book = parse_orderbook("T", LEGACY_ORDERBOOK, 0.0)
        assert book.best_yes_bid == 40
        assert book.best_yes_ask == 42

    def test_envelope_is_unwrapped_however_it_is_named(self) -> None:
        inner = LIVE_ORDERBOOK["orderbook_fp"]
        for payload in (
            LIVE_ORDERBOOK,
            {"orderbook": inner},
            inner,
        ):
            assert parse_orderbook("T", payload, 0.0).best_yes_ask == 3

    def test_empty_and_malformed_payloads_are_safe(self) -> None:
        for payload in ({}, {"orderbook_fp": {}}, {"orderbook_fp": None},
                        {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}):
            book = parse_orderbook("T", payload, 0.0)
            assert not book.is_two_sided

    def test_out_of_range_prices_are_dropped(self) -> None:
        """A settled market can quote 0.00 or 1.00, which is not a tradeable level."""
        payload = {"orderbook_fp": {"yes_dollars": [["0.0000", "10"], ["0.4000", "5"]],
                                    "no_dollars": [["1.0000", "10"]]}}
        book = parse_orderbook("T", payload, 0.0)
        assert book.best_yes_bid == 40
        assert book.best_no_bid is None


class TestLiveTrade:
    def test_current_format_parses(self) -> None:
        t = parse_trade(LIVE_TRADE)
        assert t.price == 2
        assert t.size == 21
        assert t.signed_size == 21  # taker bought YES
        assert t.ticker == "KXGDPYEAR-36-B4.8"

    def test_yes_and_no_prices_are_complementary(self) -> None:
        yes = dollars_to_cents(LIVE_TRADE["yes_price_dollars"])
        no = dollars_to_cents(LIVE_TRADE["no_price_dollars"])
        assert yes + no == PAYOUT

    def test_no_side_taker_signs_negative(self) -> None:
        payload = dict(LIVE_TRADE, taker_side="no")
        assert parse_trade(payload).signed_size == -21

    def test_legacy_trade_format_still_parses(self) -> None:
        t = parse_trade({"ticker": "T", "yes_price": 44, "count": 12,
                         "taker_side": "no", "ts": 5.0})
        assert t.price == 44
        assert t.signed_size == -12
        assert t.ts == pytest.approx(5.0)


class TestLiveMarketMeta:
    def test_between_market_parses_and_settles(self) -> None:
        meta = parse_market_meta(LIVE_MARKET)
        assert meta.strike_type == "between"
        assert meta.strike == 4.6
        assert meta.strike_upper == 5.0
        assert meta.payoff(4.8) == 1
        assert meta.payoff(5.2) == 0
        assert meta.payoff(4.5) == 0

    def test_iso_close_time_becomes_an_epoch(self) -> None:
        meta = parse_market_meta(LIVE_MARKET)
        assert meta.close_ts is not None
        assert meta.close_ts > 2_100_000_000  # 2037

    def test_missing_close_time_stays_none(self) -> None:
        payload = {k: v for k, v in LIVE_MARKET.items() if k != "close_time"}
        assert parse_market_meta(payload).close_ts is None

    def test_threshold_markets(self) -> None:
        gt = parse_market_meta({"ticker": "T", "event_ticker": "E",
                                "strike_type": "greater", "floor_strike": 6.0})
        lt = parse_market_meta({"ticker": "T", "event_ticker": "E",
                                "strike_type": "less", "cap_strike": 0.1})
        assert gt.strike_type == "gt" and gt.payoff(6.5) == 1 and gt.payoff(6.0) == 0
        assert lt.strike_type == "lt" and lt.payoff(0.05) == 1 and lt.payoff(0.1) == 0


class TestQuote:
    def test_reads_the_touch_off_a_market_payload(self) -> None:
        assert parse_quote(LIVE_MARKET) == (1, 3)

    def test_legacy_integer_fields(self) -> None:
        assert parse_quote({"yes_bid": 40, "yes_ask": 42}) == (40, 42)

    def test_missing_quote_is_none(self) -> None:
        assert parse_quote({}) == (None, None)
