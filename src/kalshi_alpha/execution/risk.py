"""Pre-trade risk.

Every order passes through here before it can reach an exchange, and the
default answer is no. The checks are ordered cheapest-first so the common
rejection costs almost nothing, and each returns a specific reason rather than
a bare boolean -- when a strategy stops trading at 2pm you need to know *which*
limit stopped it.

Two design choices worth stating:

* **The kill switch latches.** Once tripped it stays tripped until a human
  clears it. A risk system that silently re-arms itself has not stopped the
  loss, it has scheduled it.
* **Multi-leg structures are checked atomically.** An arbitrage that passes
  leg-by-leg can still breach the portfolio limit as a whole, and a half-filled
  arbitrage is not a small arbitrage -- it is an unhedged directional
  position several times the size the strategy intended.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from kalshi_alpha.config import RiskConfig
from kalshi_alpha.logging_setup import get_logger
from kalshi_alpha.types import PAYOUT, ArbOpportunity, Order, OrderBook, Position

log = get_logger(__name__)


class RiskViolation(RuntimeError):
    """Raised when an order is rejected and the caller asked for hard failure."""


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    reason: str = ""
    limit: str = ""

    def __bool__(self) -> bool:
        return self.approved


APPROVED = RiskDecision(True)


@dataclass
class RiskEngine:
    """Stateful pre-trade gate."""

    config: RiskConfig = field(default_factory=RiskConfig)
    realised_pnl_cents: float = 0.0
    peak_equity_cents: float = 0.0
    equity_cents: float = 0.0
    open_order_count: int = 0
    killed: bool = False
    kill_reason: str = ""
    rejections: dict[str, int] = field(default_factory=dict)

    # ---- lifecycle -----------------------------------------------------
    def update_equity(self, equity_cents: float) -> None:
        """Feed the current mark. Trips the kill switch on excessive drawdown."""
        self.equity_cents = float(equity_cents)
        self.peak_equity_cents = max(self.peak_equity_cents, self.equity_cents)
        drawdown = self.peak_equity_cents - self.equity_cents
        if drawdown > self.config.kill_switch_drawdown_cents:
            self.kill(
                f"drawdown {drawdown / 100:.2f} USD exceeds limit "
                f"{self.config.kill_switch_drawdown_cents / 100:.2f} USD"
            )

    def kill(self, reason: str) -> None:
        if not self.killed:
            self.killed = True
            self.kill_reason = reason
            log.error("KILL SWITCH TRIPPED", extra={"reason": reason})

    def reset_kill_switch(self) -> None:
        """Deliberately manual: a latched switch must be cleared by a person."""
        self.killed = False
        self.kill_reason = ""
        self.peak_equity_cents = self.equity_cents

    def _reject(self, limit: str, reason: str) -> RiskDecision:
        self.rejections[limit] = self.rejections.get(limit, 0) + 1
        return RiskDecision(False, reason, limit)

    # ---- checks --------------------------------------------------------
    def check_order(
        self,
        order: Order,
        positions: Mapping[str, Position],
        books: Mapping[str, OrderBook] | None = None,
    ) -> RiskDecision:
        if self.killed:
            return self._reject("kill_switch", f"kill switch active: {self.kill_reason}")
        if order.qty <= 0:
            return self._reject("qty", "order quantity must be positive")
        if order.qty > self.config.max_order_qty:
            return self._reject(
                "max_order_qty",
                f"qty {order.qty} exceeds per-order cap {self.config.max_order_qty}",
            )
        if self.open_order_count >= self.config.max_open_orders:
            return self._reject(
                "max_open_orders", f"already {self.open_order_count} open orders"
            )
        if self.realised_pnl_cents < -self.config.max_daily_loss_cents:
            return self._reject(
                "max_daily_loss",
                f"daily loss {self.realised_pnl_cents / 100:.2f} USD past limit",
            )

        pos = positions.get(order.ticker)
        current = abs(pos.yes_qty) if pos else 0
        if current + order.qty > self.config.max_contracts_per_market:
            return self._reject(
                "max_contracts_per_market",
                f"{order.ticker}: {current} + {order.qty} over "
                f"{self.config.max_contracts_per_market}",
            )

        gross = self.gross_exposure(positions, books or {})
        added = order.qty * (order.price if order.price is not None else PAYOUT)
        if gross + added > self.config.max_gross_exposure_cents:
            return self._reject(
                "max_gross_exposure",
                f"gross exposure {(gross + added) / 100:.2f} USD over "
                f"{self.config.max_gross_exposure_cents / 100:.2f} USD",
            )
        return APPROVED

    def check_arbitrage(
        self,
        opp: ArbOpportunity,
        positions: Mapping[str, Position],
        books: Mapping[str, OrderBook] | None = None,
    ) -> RiskDecision:
        """Approve a multi-leg structure as a unit, or not at all.

        A partially-filled arbitrage is an unhedged directional bet, so the
        whole structure must clear the limits before any leg is sent.
        """
        if self.killed:
            return self._reject("kill_switch", f"kill switch active: {self.kill_reason}")
        if opp.worst_case_pnl_cents <= 0:
            return self._reject(
                "not_arbitrage",
                f"worst-case P&L is {opp.worst_case_pnl_cents}c; this is not riskless",
            )
        if opp.capital_at_risk_cents > self.config.per_event_concentration_cents:
            return self._reject(
                "per_event_concentration",
                f"{opp.capital_at_risk_cents / 100:.2f} USD in {opp.event_ticker} "
                f"over concentration limit",
            )

        gross = self.gross_exposure(positions, books or {})
        added = sum(lg.qty * lg.price for lg in opp.legs)
        if gross + added > self.config.max_gross_exposure_cents:
            return self._reject(
                "max_gross_exposure",
                f"arbitrage would take gross exposure to {(gross + added) / 100:.2f} USD",
            )
        for leg in opp.legs:
            pos = positions.get(leg.ticker)
            current = abs(pos.yes_qty) if pos else 0
            if current + leg.qty > self.config.max_contracts_per_market:
                return self._reject(
                    "max_contracts_per_market",
                    f"leg {leg.ticker} would breach the per-market cap",
                )
        return APPROVED

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def gross_exposure(
        positions: Mapping[str, Position], books: Mapping[str, OrderBook]
    ) -> float:
        """Absolute marked exposure across all markets, in cents."""
        total = 0.0
        for ticker, pos in positions.items():
            if pos.yes_qty == 0:
                continue
            book = books.get(ticker)
            mark = book.mid if book and book.mid is not None else float(PAYOUT) / 2
            total += abs(pos.yes_qty) * mark
        return total

    def approve_or_raise(
        self,
        order: Order,
        positions: Mapping[str, Position],
        books: Mapping[str, OrderBook] | None = None,
    ) -> None:
        decision = self.check_order(order, positions, books)
        if not decision:
            raise RiskViolation(f"[{decision.limit}] {decision.reason}")

    def report(self) -> dict[str, object]:
        return {
            "killed": self.killed,
            "kill_reason": self.kill_reason,
            "equity_cents": self.equity_cents,
            "peak_equity_cents": self.peak_equity_cents,
            "drawdown_cents": self.peak_equity_cents - self.equity_cents,
            "open_orders": self.open_order_count,
            "rejections": dict(self.rejections),
        }


def concentration_by_event(
    positions: Mapping[str, Position],
    books: Mapping[str, OrderBook],
    event_of: Mapping[str, str],
) -> dict[str, float]:
    """Exposure grouped by event.

    Correlated markets inside one event are a single bet no matter how many
    tickers it is spread across, so this is the number the concentration limit
    should actually be applied to.
    """
    out: dict[str, float] = {}
    for ticker, pos in positions.items():
        if pos.yes_qty == 0:
            continue
        book = books.get(ticker)
        mark = book.mid if book and book.mid is not None else float(PAYOUT) / 2
        event = event_of.get(ticker, ticker)
        out[event] = out.get(event, 0.0) + abs(pos.yes_qty) * mark
    return out


def check_batch(
    engine: RiskEngine,
    orders: Sequence[Order],
    positions: Mapping[str, Position],
    books: Mapping[str, OrderBook] | None = None,
) -> tuple[list[Order], list[tuple[Order, RiskDecision]]]:
    """Filter a batch, returning approved orders and each rejection with its reason."""
    approved: list[Order] = []
    rejected: list[tuple[Order, RiskDecision]] = []
    for order in orders:
        decision = engine.check_order(order, positions, books)
        if decision:
            approved.append(order)
        else:
            rejected.append((order, decision))
    return approved, rejected
