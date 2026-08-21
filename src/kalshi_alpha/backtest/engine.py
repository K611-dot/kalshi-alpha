"""Event-driven backtester.

The loop is deliberately strict about time. At each timestamp the strategy sees
only the book as of *that* timestamp minus the market-data latency, and any
order it returns reaches the matching engine at *that* timestamp plus the order
latency. There is no path by which a decision can consume information from its
own future, which is the property that separates a backtest from a
demonstration.

Settlement is explicit rather than assumed. Positions are marked to the book
while the market is open and paid off against the realised outcome at the end,
so an arbitrage that only looks profitable because it was marked at the mid it
was executed against does not survive.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kalshi_alpha.backtest.fills import FillModel, RestingOrder
from kalshi_alpha.backtest.metrics import PerformanceStats, performance
from kalshi_alpha.config import Settings
from kalshi_alpha.logging_setup import get_logger
from kalshi_alpha.types import (
    PAYOUT,
    Action,
    Fill,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
    Trade,
)

log = get_logger(__name__)


@dataclass
class Context:
    """What a strategy is allowed to see at one point in time."""

    ts: float
    step: int
    books: Mapping[str, OrderBook]
    trades: Sequence[Trade]
    positions: Mapping[str, Position]
    cash_cents: float
    open_orders: Sequence[RestingOrder]
    settings: Settings
    state: dict = field(default_factory=dict)

    def position(self, ticker: str) -> int:
        p = self.positions.get(ticker)
        return p.yes_qty if p else 0

    def mid(self, ticker: str) -> float | None:
        book = self.books.get(ticker)
        return book.mid if book else None

    def gross_exposure_cents(self) -> float:
        total = 0.0
        for tk, pos in self.positions.items():
            mid = self.mid(tk)
            if mid is not None:
                total += abs(pos.yes_qty) * mid
        return total


@dataclass
class BacktestResult:
    equity: np.ndarray
    times: np.ndarray
    fills: pd.DataFrame
    positions: dict[str, Position]
    stats: PerformanceStats
    settled_pnl_cents: float = 0.0
    n_orders: int = 0
    n_rejected: int = 0
    diagnostics: dict[str, float] = field(default_factory=dict)

    def equity_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"ts": self.times, "equity_cents": self.equity})

    def summary(self) -> str:
        lines = [
            self.stats.summary(),
            f"  orders {self.n_orders}  rejected {self.n_rejected}  "
            f"settled P&L {self.settled_pnl_cents / 100:,.2f} USD",
        ]
        if self.diagnostics:
            lines.append(
                "  " + "  ".join(f"{k}={v:.4g}" for k, v in self.diagnostics.items())
            )
        return "\n".join(lines)


class Backtester:
    """Replays market data through a strategy with a realistic fill model."""

    def __init__(
        self,
        settings: Settings | None = None,
        fill_model: FillModel | None = None,
        starting_cash_cents: float = 1_000_000.0,
    ) -> None:
        self.settings = settings or Settings()
        self.fills = fill_model or FillModel(
            execution=self.settings.execution, fees=self.settings.fees, seed=self.settings.seed
        )
        self.starting_cash = starting_cash_cents

    def run(
        self,
        snapshots: Sequence[Mapping[str, OrderBook]],
        times: Sequence[float] | np.ndarray,
        strategy: Callable[[Context], Sequence[Order]],
        trades: Mapping[str, Sequence[Trade]] | None = None,
        settlement: Mapping[str, int] | None = None,
        n_trials: int = 1,
    ) -> BacktestResult:
        if len(snapshots) != len(times):
            raise ValueError("snapshots and times must be the same length")

        positions: dict[str, Position] = {}
        cash = float(self.starting_cash)
        resting: list[RestingOrder] = []
        pending: list[tuple[float, Order]] = []  # (arrival_ts, order)
        all_fills: list[Fill] = []
        equity: list[float] = []
        n_orders = 0
        n_rejected = 0

        trades_by_ts = _index_trades(trades or {}, times)
        exec_cfg = self.settings.execution
        risk = self.settings.risk
        order_lat = exec_cfg.order_latency_ms / 1000.0
        data_lat = exec_cfg.market_data_latency_ms / 1000.0

        state: dict = {}
        for step, (ts, books) in enumerate(zip(times, snapshots, strict=True)):
            ts = float(ts)
            tape = trades_by_ts.get(step, [])

            # 1. Orders that have now arrived at the exchange.
            still_pending: list[tuple[float, Order]] = []
            for arrival, order in pending:
                if arrival > ts:
                    still_pending.append((arrival, order))
                    continue
                book = books.get(order.ticker)
                if book is None:
                    n_rejected += 1
                    continue
                if _is_marketable(order, book):
                    fills = self.fills.fill_aggressive(order, book, ts)
                    if not fills:
                        n_rejected += 1
                    filled_qty = 0
                    for f in fills:
                        cash += f.cash_delta
                        positions.setdefault(f.ticker, Position(f.ticker)).apply(f)
                        all_fills.append(f)
                        filled_qty += f.qty
                    # A marketable GTC limit rests whatever it could not take;
                    # IOC and FOK do not. FOK is all-or-nothing, so a partial
                    # fill would be a bug, not a partial trade.
                    leftover = order.qty - filled_qty
                    if leftover > 0 and order.tif is TimeInForce.GTC and order.price is not None:
                        rest = Order(**{**order.__dict__, "qty": leftover})
                        resting.append(
                            RestingOrder(
                                order=rest,
                                submitted_ts=order.ts,
                                live_ts=ts,
                                queue_ahead=self.fills.initial_queue(rest, book),
                                status=OrderStatus.OPEN,
                            )
                        )
                elif order.tif is TimeInForce.GTC:
                    resting.append(
                        RestingOrder(
                            order=order,
                            submitted_ts=order.ts,
                            live_ts=ts,
                            queue_ahead=self.fills.initial_queue(order, book),
                            status=OrderStatus.OPEN,
                        )
                    )
                else:
                    n_rejected += 1  # IOC/FOK that arrived non-marketable
            pending = still_pending

            # 2. Passive fills from the tape, then adverse sweep-throughs.
            for ro in resting:
                for trade in tape:
                    for f in self.fills.apply_trade(ro, trade):
                        cash += f.cash_delta
                        positions.setdefault(f.ticker, Position(f.ticker)).apply(f)
                        all_fills.append(f)
                book = books.get(ro.order.ticker)
                if book is not None:
                    for f in self.fills.sweep_through(ro, book, ts):
                        cash += f.cash_delta
                        positions.setdefault(f.ticker, Position(f.ticker)).apply(f)
                        all_fills.append(f)
            resting = [ro for ro in resting if ro.is_live and ro.remaining > 0]

            # 3. Strategy decision on delayed data.
            visible = snapshots[max(0, step - int(data_lat / max(_dt(times), 1e-9)))]
            ctx = Context(
                ts=ts,
                step=step,
                books=visible,
                trades=tape,
                positions=positions,
                cash_cents=cash,
                open_orders=tuple(resting),
                settings=self.settings,
                state=state,
            )
            for order in strategy(ctx) or []:
                if not _passes_risk(order, positions, ctx, risk):
                    n_rejected += 1
                    continue
                n_orders += 1
                pending.append((ts + order_lat, order))

            # 4. Mark to market.
            equity.append(cash + _mark(positions, books))

        # 5. Settlement. Positions still open at the close are paid off against
        # the realised outcome rather than marked, so nothing is credited at a
        # mid the strategy could not actually have traded out at.
        settled = 0.0
        if settlement:
            for tk, pos in positions.items():
                if tk in settlement:
                    settled += pos.yes_qty * int(settlement[tk]) * PAYOUT + pos.no_credit_cents
                else:
                    book = snapshots[-1].get(tk) if snapshots else None
                    mid = book.mid if book else 50.0
                    settled += pos.yes_qty * (mid if mid is not None else 50.0)
                    settled += pos.no_credit_cents
            if equity:
                equity[-1] = cash + settled
            settled_pnl = cash + settled - self.starting_cash
        else:
            settled_pnl = (equity[-1] - self.starting_cash) if equity else 0.0

        fills_df = _fills_frame(all_fills)
        stats = performance(
            np.asarray(equity, dtype=float),
            fills_df,
            periods_per_year=_periods_per_year(times),
            n_trials=n_trials,
        )
        return BacktestResult(
            equity=np.asarray(equity, dtype=float),
            times=np.asarray(times, dtype=float),
            fills=fills_df,
            positions=positions,
            stats=stats,
            settled_pnl_cents=float(settled_pnl),
            n_orders=n_orders,
            n_rejected=n_rejected,
            diagnostics={
                "fill_rate": (len(all_fills) / n_orders) if n_orders else 0.0,
                "maker_share": (
                    float(np.mean([f.liquidity == "maker" for f in all_fills]))
                    if all_fills
                    else 0.0
                ),
                "unfilled_resting": float(len(resting)),
            },
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _is_marketable(order: Order, book: OrderBook) -> bool:
    """Would this order trade immediately against the book it arrived at?

    A limit order priced at or through the touch crosses; it does not rest.
    Treating every priced order as passive is a silent and expensive error: it
    hands arbitrage legs a maker fill they never earned, and the resulting
    zero-fee, front-of-queue execution flatters the strategy enormously.
    """
    if order.order_type is OrderType.MARKET or order.price is None:
        return True
    ladder = book.ladder(order.side, order.action)
    if not ladder:
        return False
    best = ladder[0].price
    if order.action is Action.BUY:
        return order.price >= best
    return order.price <= best


def _dt(times: Sequence[float] | np.ndarray) -> float:
    if len(times) < 2:
        return 1.0
    return float(np.median(np.diff(np.asarray(times, dtype=float))))


def _periods_per_year(times: Sequence[float] | np.ndarray) -> float:
    dt = _dt(times)
    return (365.25 * 24 * 3600) / dt if dt > 0 else 252.0


def _index_trades(
    trades: Mapping[str, Sequence[Trade]], times: Sequence[float] | np.ndarray
) -> dict[int, list[Trade]]:
    """Bucket every print onto the snapshot index it belongs to."""
    if not trades:
        return {}
    t_arr = np.asarray(times, dtype=float)
    out: dict[int, list[Trade]] = {}
    for tape in trades.values():
        for tr in tape:
            idx = int(np.searchsorted(t_arr, tr.ts, side="right") - 1)
            if 0 <= idx < t_arr.size:
                out.setdefault(idx, []).append(tr)
    return out


def _mark(positions: Mapping[str, Position], books: Mapping[str, OrderBook]) -> float:
    """Value of open positions at the mid, excluding cash already booked.

    Includes each position's ``no_credit_cents`` -- the unconditional
    settlement cash owed to long NO contracts -- so a fully hedged pair marks at
    its true value rather than at zero.
    """
    total = 0.0
    for tk, pos in positions.items():
        if pos.yes_qty == 0 and pos.no_credit_cents == 0:
            continue
        book = books.get(tk)
        mid = book.mid if book else None
        total += pos.yes_qty * (mid if mid is not None else 50.0)
        total += pos.no_credit_cents
    return total


def _passes_risk(order: Order, positions: Mapping[str, Position], ctx: Context, risk) -> bool:
    if order.qty <= 0 or order.qty > risk.max_order_qty:
        return False
    pos = positions.get(order.ticker)
    current = abs(pos.yes_qty) if pos else 0
    if current + order.qty > risk.max_contracts_per_market:
        return False
    return not ctx.gross_exposure_cents() > risk.max_gross_exposure_cents


def _fills_frame(fills: Sequence[Fill]) -> pd.DataFrame:
    if not fills:
        return pd.DataFrame(
            columns=["ts", "ticker", "side", "action", "qty", "price", "fee_cents",
                     "liquidity", "tag"]
        )
    return pd.DataFrame(
        {
            "ts": [f.ts for f in fills],
            "ticker": [f.ticker for f in fills],
            "side": [f.side.value for f in fills],
            "action": [f.action.value for f in fills],
            "qty": [f.qty for f in fills],
            "price": [f.price for f in fills],
            "fee_cents": [f.fee_cents for f in fills],
            "liquidity": [f.liquidity for f in fills],
            "tag": [f.tag for f in fills],
        }
    )
