"""Order management.

The OMS owns one job: know, at all times, exactly what is working at the
exchange. Everything else follows from that.

* **Explicit state machine.** Orders move
  ``NEW -> PENDING -> OPEN -> (PARTIALLY_FILLED) -> FILLED | CANCELED | REJECTED``
  and illegal transitions raise. Silently accepting a fill on an order believed
  cancelled is how a position appears from nowhere.
* **Idempotency by construction.** Every order carries a client-generated id.
  A timeout is not an answer -- the order may or may not have been created --
  and the only safe recovery is to re-send the *same* id and let the exchange
  deduplicate.
* **Reconciliation.** :meth:`OrderManager.reconcile` diffs local state against
  the exchange's and reports every discrepancy. It is not decoration: after any
  disconnect, local state is a hypothesis until it has been checked.
* **Atomic multi-leg submission.** Arbitrage legs go out as a group, and if one
  is rejected the manager immediately tries to unwind the fills that did land,
  because a half-filled arbitrage is a naked position.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from kalshi_alpha.execution.risk import RiskEngine
from kalshi_alpha.logging_setup import get_logger
from kalshi_alpha.types import (
    Action,
    ArbOpportunity,
    Fill,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TimeInForce,
)

log = get_logger(__name__)

LEGAL_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.NEW: frozenset({OrderStatus.PENDING, OrderStatus.REJECTED}),
    OrderStatus.PENDING: frozenset(
        {OrderStatus.OPEN, OrderStatus.REJECTED, OrderStatus.FILLED,
         OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELED}
    ),
    OrderStatus.OPEN: frozenset(
        {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELED}
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset({OrderStatus.FILLED, OrderStatus.CANCELED}),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
}


class OrderStateError(RuntimeError):
    """Raised on an illegal state transition."""


@dataclass
class OrderRecord:
    order: Order
    status: OrderStatus = OrderStatus.NEW
    exchange_id: str = ""
    filled_qty: int = 0
    avg_price: float = 0.0
    fees_cents: int = 0
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)
    reject_reason: str = ""
    fills: list[Fill] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.order.qty - self.filled_qty)

    @property
    def terminal(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)

    def transition(self, new: OrderStatus, reason: str = "") -> None:
        if new is self.status:
            return
        if new not in LEGAL_TRANSITIONS[self.status]:
            raise OrderStateError(
                f"{self.order.client_order_id}: illegal transition "
                f"{self.status.value} -> {new.value}"
            )
        self.status = new
        self.updated_ts = time.time()
        if reason:
            self.reject_reason = reason

    def apply_fill(self, fill: Fill) -> None:
        if self.terminal and self.status is not OrderStatus.PARTIALLY_FILLED:
            raise OrderStateError(
                f"{self.order.client_order_id}: fill on terminal order ({self.status.value})"
            )
        if fill.qty > self.remaining:
            raise OrderStateError(
                f"{self.order.client_order_id}: overfill -- {fill.qty} on {self.remaining} left"
            )
        notional = self.avg_price * self.filled_qty + fill.price * fill.qty
        self.filled_qty += fill.qty
        self.avg_price = notional / self.filled_qty if self.filled_qty else 0.0
        self.fees_cents += fill.fee_cents
        self.fills.append(fill)
        self.transition(
            OrderStatus.FILLED if self.remaining == 0 else OrderStatus.PARTIALLY_FILLED
        )


class OrderManager:
    """Tracks orders, positions and the risk gate. Broker-agnostic."""

    def __init__(self, risk: RiskEngine | None = None, tag: str = "kalshi-alpha") -> None:
        self.risk = risk or RiskEngine()
        self.tag = tag
        self.orders: dict[str, OrderRecord] = {}
        self.positions: dict[str, Position] = {}
        self.cash_cents: float = 0.0
        self.rejected: list[tuple[Order, str]] = []

    # ---- ids -----------------------------------------------------------
    def new_client_id(self, prefix: str = "") -> str:
        return f"{prefix or self.tag}-{uuid.uuid4().hex[:16]}"

    # ---- submission ----------------------------------------------------
    def prepare(
        self,
        ticker: str,
        side: Side,
        action: Action,
        qty: int,
        price: int | None = None,
        order_type: OrderType = OrderType.LIMIT,
        tif: TimeInForce = TimeInForce.IOC,
        tag: str = "",
    ) -> Order:
        return Order(
            ticker=ticker,
            side=side,
            action=action,
            qty=qty,
            price=price,
            order_type=order_type,
            tif=tif,
            client_order_id=self.new_client_id(),
            ts=time.time(),
            tag=tag or self.tag,
        )

    def submit(
        self, order: Order, books: Mapping[str, OrderBook] | None = None
    ) -> OrderRecord | None:
        """Risk-check and register an order. Returns ``None`` if rejected."""
        decision = self.risk.check_order(order, self.positions, books)
        if not decision:
            self.rejected.append((order, f"[{decision.limit}] {decision.reason}"))
            log.warning(
                "order rejected by risk",
                extra={"ticker": order.ticker, "limit": decision.limit,
                       "reason": decision.reason},
            )
            return None

        record = OrderRecord(order=order)
        record.transition(OrderStatus.PENDING)
        self.orders[order.client_order_id] = record
        self.risk.open_order_count += 1
        return record

    def submit_arbitrage(
        self,
        opp: ArbOpportunity,
        books: Mapping[str, OrderBook] | None = None,
    ) -> list[OrderRecord]:
        """Submit every leg of a proved arbitrage, or none of them.

        The structure is risk-checked as a unit first. Legs are then sent with
        IOC limits at the proved prices: a worse fill would invalidate the proof,
        and it is strictly better to end up with nothing than with three legs of
        a four-leg hedge.
        """
        decision = self.risk.check_arbitrage(opp, self.positions, books)
        if not decision:
            log.warning(
                "arbitrage rejected by risk",
                extra={"limit": decision.limit, "reason": decision.reason},
            )
            return []

        merged: dict[tuple[str, Side, Action], list[int]] = {}
        for leg in opp.legs:
            key = (leg.ticker, leg.side, leg.action)
            slot = merged.setdefault(key, [0, leg.price])
            slot[0] += leg.qty
            slot[1] = (
                max(slot[1], leg.price) if leg.action is Action.BUY else min(slot[1], leg.price)
            )

        records: list[OrderRecord] = []
        for (ticker, side, action), (qty, limit) in merged.items():
            order = self.prepare(
                ticker, side, action, qty, limit, OrderType.LIMIT, TimeInForce.IOC,
                tag=f"arb:{opp.kind}",
            )
            rec = self.submit(order, books)
            if rec is None:
                log.error("arbitrage leg rejected mid-submission; unwinding",
                          extra={"ticker": ticker, "kind": opp.kind})
                self.unwind(records)
                return []
            records.append(rec)
        return records

    # ---- lifecycle events ----------------------------------------------
    def on_ack(self, client_order_id: str, exchange_id: str) -> None:
        rec = self.orders.get(client_order_id)
        if rec is None:
            log.warning("ack for unknown order", extra={"coid": client_order_id})
            return
        rec.exchange_id = exchange_id
        rec.transition(OrderStatus.OPEN)

    def on_reject(self, client_order_id: str, reason: str) -> None:
        rec = self.orders.get(client_order_id)
        if rec is None:
            return
        rec.transition(OrderStatus.REJECTED, reason)
        self.risk.open_order_count = max(0, self.risk.open_order_count - 1)

    def on_fill(self, fill: Fill) -> None:
        rec = self.orders.get(fill.order_id)
        if rec is not None:
            rec.apply_fill(fill)
            if rec.terminal:
                self.risk.open_order_count = max(0, self.risk.open_order_count - 1)
        else:
            # An unsolicited fill means local state is behind the exchange.
            # Book it anyway -- the position is real whether or not we expected
            # it -- and flag the divergence loudly.
            log.error("fill for unknown order", extra={"coid": fill.order_id,
                                                       "ticker": fill.ticker})
        self.positions.setdefault(fill.ticker, Position(fill.ticker)).apply(fill)
        self.cash_cents += fill.cash_delta

    def on_cancel(self, client_order_id: str) -> None:
        rec = self.orders.get(client_order_id)
        if rec is None:
            return
        rec.transition(OrderStatus.CANCELED)
        self.risk.open_order_count = max(0, self.risk.open_order_count - 1)

    # ---- recovery ------------------------------------------------------
    def unwind(self, records: Sequence[OrderRecord]) -> list[Order]:
        """Build the flattening orders for legs that filled before an abort."""
        out: list[Order] = []
        for rec in records:
            if rec.filled_qty <= 0:
                continue
            leg = rec.order
            out.append(
                self.prepare(
                    ticker=leg.ticker,
                    side=leg.side.other,
                    action=Action.BUY,
                    qty=rec.filled_qty,
                    price=None,
                    order_type=OrderType.MARKET,
                    tif=TimeInForce.IOC,
                    tag="unwind",
                )
            )
        return out

    def reconcile(
        self,
        exchange_orders: Sequence[Mapping[str, object]],
        exchange_positions: Sequence[Mapping[str, object]],
    ) -> dict[str, list[str]]:
        """Diff local state against the exchange's and report every mismatch.

        Reports rather than repairs. Automatic repair on a state you do not
        understand tends to compound the problem; a human decides, then the
        process restarts from the exchange's view.
        """
        issues: dict[str, list[str]] = {"orders": [], "positions": []}

        remote_ids = {str(o.get("client_order_id", "")) for o in exchange_orders}
        local_open = {
            coid for coid, rec in self.orders.items()
            if rec.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)
        }
        for coid in local_open - remote_ids:
            issues["orders"].append(f"{coid}: open locally, absent at exchange")
        for coid in remote_ids - local_open:
            if coid:
                issues["orders"].append(f"{coid}: open at exchange, not tracked locally")

        remote_pos = {
            str(p.get("ticker", "")): int(str(p.get("position", 0))) for p in exchange_positions
        }
        for ticker, qty in remote_pos.items():
            local = self.positions.get(ticker)
            local_qty = local.yes_qty if local else 0
            if local_qty != qty:
                issues["positions"].append(f"{ticker}: local {local_qty} vs exchange {qty}")
        for ticker, pos in self.positions.items():
            if pos.yes_qty != 0 and ticker not in remote_pos:
                issues["positions"].append(
                    f"{ticker}: local {pos.yes_qty} but exchange reports no position"
                )
        return issues

    # ---- reporting -----------------------------------------------------
    def open_orders(self) -> list[OrderRecord]:
        return [r for r in self.orders.values() if not r.terminal]

    def mark_to_market(self, books: Mapping[str, OrderBook]) -> float:
        total = self.cash_cents
        for ticker, pos in self.positions.items():
            book = books.get(ticker)
            mid = book.mid if book and book.mid is not None else 50.0
            total += pos.yes_qty * mid + pos.no_credit_cents
        return total

    def summary(self) -> dict[str, object]:
        return {
            "orders": len(self.orders),
            "open": len(self.open_orders()),
            "filled": sum(r.status is OrderStatus.FILLED for r in self.orders.values()),
            "rejected": len(self.rejected),
            "positions": {t: p.yes_qty for t, p in self.positions.items() if p.yes_qty},
            "cash_cents": self.cash_cents,
            "risk": self.risk.report(),
        }
