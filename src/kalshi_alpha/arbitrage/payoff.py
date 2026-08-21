"""State-space payoff evaluation.

Every arbitrage claim in this package is *proved*, not asserted. A detector
proposes a portfolio of legs; this module enumerates the settlement states the
portfolio can end in, computes realised P&L in each one, and only lets the
opportunity through if the **worst** state is strictly profitable.

That discipline is what separates a real arbitrage from a mispricing bet: a
mispricing has positive expected value under a model, an arbitrage has positive
value under *every* state of the world, with no model at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from kalshi_alpha.types import PAYOUT, Action, ArbOpportunity, Leg, Side


def leg_terminal_cents(leg: Leg, settles_yes: bool) -> int:
    """Settlement cash received by one leg, in cents.

    A long YES position collects 100c per contract if the market settles YES;
    a long NO position collects 100c if it settles NO. Short legs are the
    negative of the corresponding long. Entry cash flow is handled separately
    by :attr:`Leg.cash_out`.
    """
    signed = leg.qty * (1 if leg.action is Action.BUY else -1)
    if leg.side is Side.YES:
        return signed * PAYOUT if settles_yes else 0
    return 0 if settles_yes else signed * PAYOUT


def portfolio_cost(legs: Sequence[Leg]) -> int:
    """Net cash out to establish the portfolio (negative = net credit)."""
    return sum(leg.cash_out for leg in legs)


def state_pnl(legs: Sequence[Leg], winner: str | None) -> int:
    """P&L in cents in the state where ``winner`` settles YES and all others NO.

    ``winner=None`` represents the state in which *no* market in the portfolio
    settles YES (only reachable for non-exhaustive groups).
    """
    receipts = sum(leg_terminal_cents(leg, leg.ticker == winner) for leg in legs)
    return receipts - portfolio_cost(legs)


def pnl_over_states(legs: Sequence[Leg], states: Sequence[str | None]) -> np.ndarray:
    return np.array([state_pnl(legs, s) for s in states], dtype=float)


def pnl_over_outcomes(
    legs: Sequence[Leg],
    settle_map: Mapping[str, Sequence[int]],
) -> np.ndarray:
    """General evaluator for arbitrary payoff structures.

    ``settle_map[ticker][j]`` is 1 if that market settles YES in state ``j``.
    Used for strike ladders, where several markets settle YES simultaneously
    and the mutually-exclusive assumption does not apply.
    """
    tickers = list(settle_map)
    n_states = len(next(iter(settle_map.values()))) if tickers else 0
    if any(len(settle_map[t]) != n_states for t in tickers):
        raise ValueError("all tickers must define the same number of settlement states")

    cost = portfolio_cost(legs)

    # Precompute each leg's signed YES-equivalent exposure once. The inner loop
    # then costs one multiply-add per (leg, state) with no attribute lookups --
    # this function is called tens of thousands of times per scan, so the
    # difference between this and a naive re-derivation is milliseconds of
    # scanner latency.
    exposures = []
    for leg in legs:
        row = settle_map.get(leg.ticker)
        if row is None:
            raise KeyError(f"no settlement row for {leg.ticker}")
        signed = leg.qty * (1 if leg.action is Action.BUY else -1)
        if leg.side is Side.YES:
            exposures.append((row, signed * PAYOUT, 0))
        else:
            exposures.append((row, 0, signed * PAYOUT))

    out = np.empty(n_states, dtype=float)
    for j in range(n_states):
        receipts = 0
        for row, yes_val, no_val in exposures:
            receipts += yes_val if row[j] else no_val
        out[j] = receipts - cost
    return out


def ladder_settle_map(
    tickers: Sequence[str], strikes: Sequence[float], direction: str = "gte"
) -> dict[str, list[int]]:
    """Settlement matrix for a monotone strike ladder.

    With ``k`` strikes the underlying can land in ``k + 1`` regions; region
    ``j`` means ``strikes[j-1] <= X < strikes[j]``. For a ``gte`` ladder,
    market ``i`` settles YES in region ``j`` iff ``j > i``.
    """
    if len(tickers) != len(strikes):
        raise ValueError("tickers and strikes must be the same length")
    k = len(tickers)
    out: dict[str, list[int]] = {}
    for i, tk in enumerate(tickers):
        if direction in ("gte", "gt"):
            out[tk] = [1 if j > i else 0 for j in range(k + 1)]
        else:  # lte / lt ladders are the mirror image
            out[tk] = [1 if j <= i else 0 for j in range(k + 1)]
    return out


def exclusive_settle_map(tickers: Sequence[str], exhaustive: bool = True) -> dict[str, list[int]]:
    """Settlement matrix for a mutually exclusive group.

    Adds a trailing "none of the above" state when the group is not exhaustive.
    """
    n = len(tickers)
    n_states = n if exhaustive else n + 1
    return {tk: [1 if j == i else 0 for j in range(n_states)] for i, tk in enumerate(tickers)}


def build_opportunity(
    kind: str,
    event_ticker: str,
    legs: Sequence[Leg],
    settle_map: Mapping[str, Sequence[int]],
    ts: float = 0.0,
    detail: str = "",
) -> ArbOpportunity | None:
    """Validate a proposed portfolio and wrap it as an opportunity.

    Returns ``None`` unless the worst settlement state is strictly profitable.
    """
    if not legs:
        return None
    pnl = pnl_over_outcomes(legs, settle_map)
    worst = int(np.floor(pnl.min()))
    if worst <= 0:
        return None
    return ArbOpportunity(
        kind=kind,
        event_ticker=event_ticker,
        legs=tuple(legs),
        cost_cents=portfolio_cost(legs),
        worst_case_pnl_cents=worst,
        best_case_pnl_cents=int(np.floor(pnl.max())),
        ts=ts,
        detail=detail,
    )


def verify_opportunity(opp: ArbOpportunity, settle_map: Mapping[str, Sequence[int]]) -> bool:
    """Independent re-check used by the tests and the pre-trade risk gate."""
    pnl = pnl_over_outcomes(opp.legs, settle_map)
    return bool(pnl.min() > 0)
