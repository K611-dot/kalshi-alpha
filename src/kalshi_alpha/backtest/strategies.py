"""Strategies.

Four, ordered by how much they assume:

1. :class:`LadderArbStrategy` -- assumes nothing. Executes only portfolios the
   arbitrage engine has *proved* profitable in every settlement state. Its P&L
   is the honest measure of how much model-free edge the book leaves on the
   table after fees, and it is the benchmark the others must beat to justify
   taking risk.
2. :class:`CoherenceStrategy` -- assumes only internal consistency. Trades the
   residual from projecting the quoted surface onto the coherent set. Directional
   and can lose, but the signal is derived from arbitrage logic rather than from
   a forecast of the world.
3. :class:`MicropriceStrategy` -- assumes short-horizon microstructure. Uses
   order-book imbalance to predict the next few ticks. Pure microstructure,
   no view on the event.
4. :class:`DriftStrategy` -- assumes the diffusion result. Buys continuation
   after a scheduled release and holds for the measured ``t90``. This is the
   strategy that only exists because the diffusion study was done first.

Every one of them is fee-aware before it emits an order. On this exchange the
round-trip taker fee at even money is 3.5 cents, which is wider than most of
the edges that look attractive on a chart, so a signal that is not compared
against :func:`~kalshi_alpha.arbitrage.fees.round_trip_breakeven_cents` is not
a signal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from kalshi_alpha.arbitrage.engine import ArbEngine
from kalshi_alpha.arbitrage.fees import round_trip_breakeven_cents
from kalshi_alpha.backtest.engine import Context
from kalshi_alpha.probability.constraints import coherent_mispricing_signal
from kalshi_alpha.types import (
    PAYOUT,
    Action,
    EventGroup,
    LadderGroup,
    Order,
    OrderType,
    Side,
    TimeInForce,
)


class Strategy(Protocol):
    """Anything callable with a :class:`Context` that returns orders."""

    def __call__(self, ctx: Context) -> Sequence[Order]: ...


def _oid(prefix: str, step: int, n: int) -> str:
    return f"{prefix}-{step:07d}-{n:02d}"


# --------------------------------------------------------------------------
@dataclass
class LadderArbStrategy:
    """Execute proved arbitrages, and nothing else."""

    ladders: Sequence[LadderGroup] = ()
    groups: Sequence[EventGroup] = ()
    scan_every: int = 5
    max_qty: int = 200
    use_lp: bool = False
    engine: ArbEngine | None = None
    name: str = "ladder_arb"
    executed: int = field(default=0, init=False)

    def __call__(self, ctx: Context) -> list[Order]:
        if ctx.step % self.scan_every:
            return []
        engine = self.engine or ArbEngine(ctx.settings)
        result = engine.scan(ctx.books, self.groups, self.ladders, use_lp=self.use_lp)
        if not result.opportunities:
            return []

        best = result.opportunities[0]

        # The detector reports one leg per price level consumed. Sending those
        # as separate orders would try to take the same displayed size twice,
        # so collapse them per (ticker, side) into a single order whose limit is
        # the worst price the detector was willing to pay.
        merged: dict[tuple[str, Side, Action], list[int]] = {}
        for leg in best.legs:
            key = (leg.ticker, leg.side, leg.action)
            slot = merged.setdefault(key, [0, leg.price])
            slot[0] += leg.qty
            slot[1] = (
                max(slot[1], leg.price) if leg.action is Action.BUY else min(slot[1], leg.price)
            )

        orders: list[Order] = []
        for i, ((ticker, side, action), (qty, limit)) in enumerate(merged.items()):
            qty = min(qty, self.max_qty)
            if qty <= 0:
                continue
            orders.append(
                Order(
                    ticker=ticker,
                    side=side,
                    action=action,
                    qty=qty,
                    # Cross with a limit at the proved price. The arbitrage was
                    # proved at *these* prices, so a worse fill can turn a proof
                    # into a loss -- better to miss the trade than to leg into it.
                    price=limit,
                    order_type=OrderType.LIMIT,
                    tif=TimeInForce.IOC,
                    client_order_id=_oid(self.name, ctx.step, i),
                    ts=ctx.ts,
                    tag=f"{self.name}:{best.kind}",
                )
            )
        if orders:
            self.executed += 1
        return orders


# --------------------------------------------------------------------------
@dataclass
class CoherenceStrategy:
    """Trade the residual from projecting the ladder onto the monotone cone."""

    ladder: LadderGroup | None = None
    entry_cents: float = 2.5
    exit_cents: float = 0.5
    qty: int = 20
    max_position: int = 100
    scan_every: int = 4
    name: str = "coherence"

    def __call__(self, ctx: Context) -> list[Order]:
        if self.ladder is None or ctx.step % self.scan_every:
            return []
        probs: dict[str, float] = {}
        for tk in self.ladder.tickers:
            book = ctx.books.get(tk)
            if book is None or not book.is_two_sided:
                return []
            mid = book.mid
            if mid is None:
                return []
            probs[tk] = mid / PAYOUT

        signal = coherent_mispricing_signal(
            probs, kind="ladder", decreasing=self.ladder.decreasing,
            order=list(self.ladder.tickers)
        )
        orders: list[Order] = []
        for i, (tk, resid) in enumerate(signal.items()):
            cents = resid * PAYOUT
            book = ctx.books[tk]
            mid = book.mid or 50.0
            hurdle = max(self.entry_cents, round_trip_breakeven_cents(int(round(mid))))
            held = ctx.position(tk)

            if abs(cents) < self.exit_cents and held != 0:
                # Flatten by taking the opposite side of the position.
                side = Side.NO if held > 0 else Side.YES
                orders.append(
                    Order(
                        ticker=tk, side=side, action=Action.BUY, qty=abs(held),
                        price=None, order_type=OrderType.MARKET, tif=TimeInForce.IOC,
                        client_order_id=_oid(self.name + "x", ctx.step, i), ts=ctx.ts,
                        tag=f"{self.name}:exit",
                    )
                )
                continue

            if abs(cents) < hurdle or abs(held) >= self.max_position:
                continue
            # Quoted rich relative to the coherent surface -> buy NO.
            side = Side.NO if cents > 0 else Side.YES
            orders.append(
                Order(
                    ticker=tk, side=side, action=Action.BUY, qty=self.qty,
                    price=None, order_type=OrderType.MARKET, tif=TimeInForce.IOC,
                    client_order_id=_oid(self.name, ctx.step, i), ts=ctx.ts,
                    tag=f"{self.name}:entry",
                )
            )
        return orders


# --------------------------------------------------------------------------
@dataclass
class MicropriceStrategy:
    """Lean on order-book imbalance over a few-tick horizon."""

    tickers: Sequence[str] = ()
    threshold: float = 0.30
    qty: int = 10
    max_position: int = 60
    hold_steps: int = 12
    name: str = "microprice"

    def __call__(self, ctx: Context) -> list[Order]:
        entries: dict[str, int] = ctx.state.setdefault("micro_entry", {})
        orders: list[Order] = []
        for i, tk in enumerate(self.tickers or list(ctx.books)):
            book = ctx.books.get(tk)
            if book is None or not book.is_two_sided:
                continue
            mid, micro = book.mid, book.microprice
            if mid is None or micro is None or not book.spread:
                continue
            held = ctx.position(tk)

            if held != 0 and ctx.step - entries.get(tk, ctx.step) >= self.hold_steps:
                side = Side.NO if held > 0 else Side.YES
                orders.append(
                    Order(ticker=tk, side=side, action=Action.BUY, qty=abs(held), price=None,
                          order_type=OrderType.MARKET, tif=TimeInForce.IOC,
                          client_order_id=_oid(self.name + "x", ctx.step, i), ts=ctx.ts,
                          tag=f"{self.name}:exit")
                )
                entries.pop(tk, None)
                continue

            tilt = (micro - mid) / book.spread
            # The microprice must lean by more than the fee costs to cross.
            edge_cents = abs(tilt) * book.spread
            if edge_cents < round_trip_breakeven_cents(int(round(mid))):
                continue
            if abs(tilt) < self.threshold or abs(held) >= self.max_position:
                continue
            side = Side.YES if tilt > 0 else Side.NO
            orders.append(
                Order(ticker=tk, side=side, action=Action.BUY, qty=self.qty, price=None,
                      order_type=OrderType.MARKET, tif=TimeInForce.IOC,
                      client_order_id=_oid(self.name, ctx.step, i), ts=ctx.ts,
                      tag=f"{self.name}:entry")
            )
            entries[tk] = ctx.step
        return orders


# --------------------------------------------------------------------------
@dataclass
class DriftStrategy:
    """Trade post-release continuation, sized off the measured diffusion speed.

    Entry waits ``entry_delay_s`` after the release -- long enough for the jump
    itself to print, since the jump is not the trade. The trade is the *drift
    that follows it*, which exists only because the market underreacts. The
    hold is ``t90`` from the diffusion study: exit once the measured adjustment
    is essentially complete, because everything after that is pure risk with no
    expected return.
    """

    tickers: Sequence[str] = ()
    event_ts: Sequence[float] = ()
    entry_delay_s: float = 30.0
    hold_s: float = 600.0
    lookback_s: float = 120.0
    min_move_cents: float = 2.0
    qty: int = 25
    name: str = "drift"

    def __call__(self, ctx: Context) -> list[Order]:
        hist: dict[str, list[tuple[float, float]]] = ctx.state.setdefault("drift_hist", {})
        entries: dict[str, float] = ctx.state.setdefault("drift_entry", {})
        orders: list[Order] = []

        for i, tk in enumerate(self.tickers or list(ctx.books)):
            book = ctx.books.get(tk)
            if book is None:
                continue
            mid = book.mid
            if mid is None:
                continue
            series = hist.setdefault(tk, [])
            series.append((ctx.ts, mid))
            if len(series) > 4_000:
                del series[:-4_000]

            # Exit once the measured adjustment window has elapsed.
            held = ctx.position(tk)
            if held != 0 and ctx.ts - entries.get(tk, ctx.ts) >= self.hold_s:
                side = Side.NO if held > 0 else Side.YES
                orders.append(
                    Order(ticker=tk, side=side, action=Action.BUY, qty=abs(held), price=None,
                          order_type=OrderType.MARKET, tif=TimeInForce.IOC,
                          client_order_id=_oid(self.name + "x", ctx.step, i), ts=ctx.ts,
                          tag=f"{self.name}:exit")
                )
                entries.pop(tk, None)
                continue
            if held != 0:
                continue

            recent_event = [
                e for e in self.event_ts
                if 0 < ctx.ts - e <= self.entry_delay_s + max(1.0, self.lookback_s * 0.25)
            ]
            if not recent_event or ctx.ts - max(recent_event) < self.entry_delay_s:
                continue

            evt = max(recent_event)
            pre = [p for t, p in series if evt - self.lookback_s <= t < evt]
            post = [p for t, p in series if t >= evt]
            if len(pre) < 3 or len(post) < 2:
                continue
            move = float(np.mean(post[-3:]) - np.mean(pre[-3:]))
            if abs(move) < max(self.min_move_cents, round_trip_breakeven_cents(int(round(mid)))):
                continue

            # Continuation: follow the direction of the initial repricing.
            side = Side.YES if move > 0 else Side.NO
            orders.append(
                Order(ticker=tk, side=side, action=Action.BUY, qty=self.qty, price=None,
                      order_type=OrderType.MARKET, tif=TimeInForce.IOC,
                      client_order_id=_oid(self.name, ctx.step, i), ts=ctx.ts,
                      tag=f"{self.name}:entry")
            )
            entries[tk] = ctx.ts
        return orders


# --------------------------------------------------------------------------
def build_strategy(name: str, **kwargs) -> Callable[[Context], Sequence[Order]]:
    """Factory used by the CLI."""
    table = {
        "ladder_arb": LadderArbStrategy,
        "coherence": CoherenceStrategy,
        "microprice": MicropriceStrategy,
        "drift": DriftStrategy,
    }
    if name not in table:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(table)}")
    cls = table[name]
    fields = getattr(cls, "__dataclass_fields__", {})
    valid = {k: v for k, v in kwargs.items() if k in fields}
    return cls(**valid)


STRATEGY_NAMES = ("ladder_arb", "coherence", "microprice", "drift")
