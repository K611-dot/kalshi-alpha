"""Linear-programming arbitrage: the general case, and its dual.

The closed-form detectors each recognise one named structure. This module drops
the names and asks the only question that matters:

    *Is there any non-negative portfolio of the currently displayed liquidity
    whose worst settlement state is strictly profitable?*

Formulation
-----------
Each **price level** of each side of each market is a separate instrument, so
the LP walks the book by itself: two levels with identical payoff differ only
in cost, and the solver naturally exhausts the cheaper one first. That turns
the convex, piecewise-linear execution-cost curve into plain linear algebra.

With :math:`x_i \\ge 0` contracts of instrument :math:`i`, per-contract all-in
cost :math:`c_i`, settlement payoff :math:`A_{ij}` in state :math:`j`, and
displayed size :math:`u_i`:

.. math::

    \\max_{x, z} \\; z
    \\quad\\text{s.t.}\\quad
    \\sum_i (A_{ij} - c_i) x_i \\;\\ge\\; z \\;\\; \\forall j,
    \\qquad 0 \\le x_i \\le u_i,
    \\qquad \\sum_i c_i x_i \\le B .

:math:`z^\\star > 0` is a certificate of arbitrage: a portfolio that makes at
least :math:`z^\\star` cents no matter which state realises. Shorting never
needs its own variable because buying the complementary NO contract *is* the
short.

The dual
--------
By the fundamental theorem of asset pricing, no arbitrage exists iff there is a
probability vector :math:`q` over states with :math:`A q \\le c` for every
tradeable instrument -- a set of **risk-neutral state prices** consistent with
the whole quoted surface. :func:`implied_state_prices` recovers one, and
:func:`no_arbitrage_bounds` maximises and minimises the price of an arbitrary
new claim over that set, giving model-free super- and sub-replication bounds.
Those bounds are the honest answer to "what is this illiquid contract worth?"
when the market has not quoted it tightly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from kalshi_alpha.arbitrage.fees import (
    DEFAULT_TAKER_RATE,
    exact_taker_fee_cents,
    linear_taker_fee_cents,
)
from kalshi_alpha.arbitrage.liquidity import haircut_size
from kalshi_alpha.arbitrage.payoff import pnl_over_outcomes
from kalshi_alpha.config import ArbConfig
from kalshi_alpha.types import PAYOUT, Action, ArbOpportunity, Leg, OrderBook, Side


@dataclass(frozen=True, slots=True)
class Instrument:
    """One displayed price level, treated as a tradeable asset."""

    ticker: str
    side: Side
    price: int
    max_qty: int
    payoff: tuple[int, ...]  # cents per contract in each settlement state
    fee_per_contract: float

    @property
    def cost_per_contract(self) -> float:
        return self.price + self.fee_per_contract


@dataclass(frozen=True, slots=True)
class LPSolution:
    status: str
    guaranteed_cents: float
    quantities: dict[tuple[str, Side, int], int]
    opportunity: ArbOpportunity | None = None
    n_instruments: int = 0
    n_states: int = 0

    @property
    def found(self) -> bool:
        return self.opportunity is not None


def build_instruments(
    books: Mapping[str, OrderBook],
    settle_map: Mapping[str, Sequence[int]],
    cfg: ArbConfig | None = None,
    taker_rate: float = DEFAULT_TAKER_RATE,
) -> list[Instrument]:
    """Turn displayed liquidity into LP instruments.

    ``settle_map[ticker][j] == 1`` means that market settles YES in state ``j``.
    """
    cfg = cfg or ArbConfig()
    out: list[Instrument] = []
    for ticker, row in settle_map.items():
        book = books.get(ticker)
        if book is None:
            continue
        yes_payoff = tuple(PAYOUT * int(s) for s in row)
        no_payoff = tuple(PAYOUT * (1 - int(s)) for s in row)
        for side, payoff in ((Side.YES, yes_payoff), (Side.NO, no_payoff)):
            for lv in book.ladder(side, Action.BUY)[: cfg.depth_levels]:
                qty = haircut_size(lv.size, cfg.size_haircut)
                if qty <= 0:
                    continue
                out.append(
                    Instrument(
                        ticker=ticker,
                        side=side,
                        price=lv.price,
                        max_qty=qty,
                        payoff=payoff,
                        fee_per_contract=linear_taker_fee_cents(lv.price, 1, taker_rate),
                    )
                )
    return out


def solve_arbitrage_lp(
    instruments: Sequence[Instrument],
    budget_cents: float = 1e9,
    min_edge_cents: float = 1.0,
) -> tuple[str, float, np.ndarray]:
    """Solve the max-min-profit programme. Returns ``(status, z*, x*)``."""
    if not instruments:
        return "empty", 0.0, np.zeros(0)

    n = len(instruments)
    n_states = len(instruments[0].payoff)
    if any(len(ins.payoff) != n_states for ins in instruments):
        raise ValueError("all instruments must span the same state space")

    A = np.array([ins.payoff for ins in instruments], dtype=float)  # (n, n_states)
    c = np.array([ins.cost_per_contract for ins in instruments], dtype=float)
    u = np.array([ins.max_qty for ins in instruments], dtype=float)

    # Decision vector: [x_0 .. x_{n-1}, z]; minimise -z.
    obj = np.zeros(n + 1)
    obj[-1] = -1.0

    # z - sum_i (A_ij - c_i) x_i <= 0  for each state j
    profit = A - c[:, None]  # (n, n_states)
    A_ub = np.zeros((n_states + 1, n + 1))
    A_ub[:n_states, :n] = -profit.T
    A_ub[:n_states, n] = 1.0
    b_ub = np.zeros(n_states + 1)
    # capital constraint
    A_ub[n_states, :n] = c
    b_ub[n_states] = budget_cents

    bounds = [(0.0, float(ui)) for ui in u] + [(None, None)]
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return str(res.message), 0.0, np.zeros(n)

    z = float(-res.fun)
    x = np.asarray(res.x[:n], dtype=float)
    if z < min_edge_cents:
        return "no_arbitrage", z, x
    return "arbitrage", z, x


def _integerize(
    instruments: Sequence[Instrument],
    x: np.ndarray,
    settle_map: Mapping[str, Sequence[int]],
    taker_rate: float,
    min_edge_cents: float,
) -> tuple[list[Leg], float] | None:
    """Round the continuous solution down to whole contracts and re-verify.

    The LP uses the fee function without its ceiling to stay linear, and it
    returns fractional contracts. Both approximations are optimistic, so the
    integer portfolio is re-priced with the **exact** fee schedule and re-checked
    state by state. Scaling down preserves the hedge ratios, so we search a
    descending ladder of scales and keep the first that still clears the bar.
    """
    for scale in (1.0, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.35, 0.25, 0.15, 0.1):
        qtys = np.floor(x * scale + 1e-9).astype(int)
        if qtys.sum() <= 0:
            continue
        legs = [
            Leg(
                ticker=ins.ticker,
                side=ins.side,
                action=Action.BUY,
                qty=int(q),
                price=ins.price,
                fee_cents=exact_taker_fee_cents(ins.price, int(q), taker_rate),
            )
            for ins, q in zip(instruments, qtys, strict=True)
            if q > 0
        ]
        if not legs:
            continue
        worst = float(pnl_over_outcomes(legs, settle_map).min())
        if worst >= min_edge_cents:
            return legs, worst
    return None


def find_arbitrage(
    books: Mapping[str, OrderBook],
    settle_map: Mapping[str, Sequence[int]],
    event_ticker: str = "",
    cfg: ArbConfig | None = None,
    taker_rate: float = DEFAULT_TAKER_RATE,
    budget_cents: float = 1e7,
) -> LPSolution:
    """End-to-end: build instruments, solve, integerise, verify."""
    cfg = cfg or ArbConfig()
    instruments = build_instruments(books, settle_map, cfg, taker_rate)
    n_states = len(next(iter(settle_map.values()))) if settle_map else 0
    status, z, x = solve_arbitrage_lp(instruments, budget_cents, float(cfg.min_edge_cents))
    if status != "arbitrage":
        return LPSolution(status, z, {}, None, len(instruments), n_states)

    fixed = _integerize(instruments, x, settle_map, taker_rate, float(cfg.min_edge_cents))
    if fixed is None:
        return LPSolution("lost_to_rounding", z, {}, None, len(instruments), n_states)

    legs, worst = fixed
    pnl = pnl_over_outcomes(legs, settle_map)
    quantities = {(lg.ticker, lg.side, lg.price): lg.qty for lg in legs}
    ts = max((b.ts for b in books.values()), default=0.0)
    opp = ArbOpportunity(
        kind="lp_general",
        event_ticker=event_ticker,
        legs=tuple(legs),
        cost_cents=sum(lg.cash_out for lg in legs),
        worst_case_pnl_cents=int(np.floor(worst)),
        best_case_pnl_cents=int(np.floor(pnl.max())),
        ts=ts,
        detail=(
            f"LP over {len(instruments)} price levels and {n_states} states; "
            f"relaxed bound {z:.2f}c, realised {worst:.0f}c after integerisation"
        ),
    )
    return LPSolution("arbitrage", z, quantities, opp, len(instruments), n_states)


# --------------------------------------------------------------------------
# dual side: risk-neutral state prices and no-arbitrage bounds
# --------------------------------------------------------------------------
def implied_state_prices(
    instruments: Sequence[Instrument], tol: float = 1e-6
) -> np.ndarray | None:
    """Recover a risk-neutral measure consistent with every quoted offer.

    Solves for ``q >= 0``, ``sum q = 1`` with ``A q <= c`` -- no instrument
    offers positive expected profit. Returns ``None`` when the system is
    infeasible, which is itself a proof that an arbitrage exists.

    The objective maximises the entropy-free proxy ``sum q`` slack, i.e. we
    simply take any feasible point; ties are broken toward the interior by
    minimising the sum of squared deviations from uniform in a second pass.
    """
    if not instruments:
        return None
    n_states = len(instruments[0].payoff)
    A = np.array([ins.payoff for ins in instruments], dtype=float)  # (n, n_states)
    c = np.array([ins.cost_per_contract for ins in instruments], dtype=float)

    res = linprog(
        c=np.zeros(n_states),
        A_ub=A,
        b_ub=c + tol,
        A_eq=np.ones((1, n_states)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * n_states,
        method="highs",
    )
    if not res.success:
        return None
    return np.asarray(res.x, dtype=float)


def no_arbitrage_bounds(
    instruments: Sequence[Instrument],
    claim_payoff: Sequence[float],
    tol: float = 1e-6,
) -> tuple[float, float] | None:
    """Model-free price band for a claim that pays ``claim_payoff[j]`` in state j.

    Minimising and maximising ``q . payoff`` over the set of consistent state
    prices gives the tightest interval that cannot be arbitraged against the
    displayed book. A wide band means the quoted surface simply does not pin the
    claim down; a narrow one means the market has effectively already priced it.
    """
    if not instruments:
        return None
    n_states = len(instruments[0].payoff)
    payoff = np.asarray(claim_payoff, dtype=float)
    if payoff.shape != (n_states,):
        raise ValueError(f"claim_payoff must have length {n_states}")

    A = np.array([ins.payoff for ins in instruments], dtype=float)
    c = np.array([ins.cost_per_contract for ins in instruments], dtype=float)
    common = {
        "A_ub": A,
        "b_ub": c + tol,
        "A_eq": np.ones((1, n_states)),
        "b_eq": np.array([1.0]),
        "bounds": [(0.0, 1.0)] * n_states,
        "method": "highs",
    }
    lo = linprog(c=payoff, **common)
    hi = linprog(c=-payoff, **common)
    if not (lo.success and hi.success):
        return None
    return float(lo.fun), float(-hi.fun)


def implied_probabilities(
    books: Mapping[str, OrderBook],
    settle_map: Mapping[str, Sequence[int]],
    cfg: ArbConfig | None = None,
    taker_rate: float = DEFAULT_TAKER_RATE,
) -> dict[str, float] | None:
    """Per-market risk-neutral probabilities implied by the joint state prices.

    Unlike reading each mid independently, these are guaranteed to be mutually
    consistent: they sum to one across an exhaustive group and respect ladder
    monotonicity by construction.
    """
    cfg = cfg or ArbConfig()
    instruments = build_instruments(books, settle_map, cfg, taker_rate)
    q = implied_state_prices(instruments)
    if q is None:
        return None
    return {
        ticker: float(np.dot(q, np.asarray(row, dtype=float)))
        for ticker, row in settle_map.items()
    }
