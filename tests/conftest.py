"""Shared fixtures."""

from __future__ import annotations

import pytest

from kalshi_alpha.config import Settings
from kalshi_alpha.data.synthetic import SimConfig, simulate_ladder
from kalshi_alpha.types import EventGroup, LadderGroup, OrderBook


def book(ticker: str, yes_bid: int, yes_ask: int, size: int = 200, depth: int = 3,
         ts: float = 0.0) -> OrderBook:
    """Build a clean two-sided book with the given touch."""
    yes_bids = [(yes_bid - k, size) for k in range(depth) if yes_bid - k > 0]
    no_bids = [((100 - yes_ask) - k, size) for k in range(depth) if (100 - yes_ask) - k > 0]
    return OrderBook.from_levels(ticker, ts, yes_bids, no_bids)


@pytest.fixture
def mk_book():
    return book


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def sim():
    """A small simulated session, shared across the suite for speed."""
    cfg = SimConfig(
        n_steps=600, dt_s=5.0, event_step=200, adjustment_half_life_s=120.0,
        event_jump_cents=18.0, seed=17,
    )
    return simulate_ladder(cfg=cfg)


@pytest.fixture
def exhaustive_group() -> EventGroup:
    return EventGroup("EVT", ("A", "B", "C"), exhaustive=True)


@pytest.fixture
def ladder() -> LadderGroup:
    return LadderGroup("CPI", ("K1", "K2", "K3"), (2.0, 3.0, 4.0), "gte")
