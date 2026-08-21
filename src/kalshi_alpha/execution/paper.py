"""Paper broker.

Implements the same interface the live client exposes, so the strategy code
that runs against production runs unchanged here. That symmetry is the point:
a paper mode that takes a different code path proves nothing about the code
path that will actually trade.

Fills reuse :class:`~kalshi_alpha.backtest.fills.FillModel`, so paper trading
and backtesting agree by construction rather than by coincidence -- a
divergence between them would otherwise be indistinguishable from alpha.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from kalshi_alpha.backtest.fills import FillModel
from kalshi_alpha.config import Settings
from kalshi_alpha.execution.oms import OrderManager
from kalshi_alpha.logging_setup import get_logger
from kalshi_alpha.types import Fill, Order, OrderBook, Position

log = get_logger(__name__)


@dataclass
class PaperBroker:
    """Simulated exchange that fills against a real (or replayed) book."""

    settings: Settings = field(default_factory=Settings)
    starting_cash_cents: int = 1_000_000
    fill_model: FillModel | None = None
    oms: OrderManager | None = None
    _seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.fill_model is None:
            self.fill_model = FillModel(
                execution=self.settings.execution,
                fees=self.settings.fees,
                seed=self.settings.seed,
            )
        if self.oms is None:
            self.oms = OrderManager()
        self.oms.cash_cents = float(self.starting_cash_cents)

    # ---- exchange surface ----------------------------------------------
    def place(self, order: Order, books: Mapping[str, OrderBook]) -> list[Fill]:
        """Risk-check, acknowledge, and fill against the current book."""
        assert self.oms is not None and self.fill_model is not None
        record = self.oms.submit(order, books)
        if record is None:
            return []

        self._seq += 1
        self.oms.on_ack(order.client_order_id, f"paper-{self._seq:08d}")

        book = books.get(order.ticker)
        if book is None:
            self.oms.on_cancel(order.client_order_id)
            return []

        fills = self.fill_model.fill_aggressive(order, book, time.time())
        for f in fills:
            self.oms.on_fill(f)
        if record.remaining > 0:
            # Everything the paper broker sends is marketable; anything left
            # over did not find liquidity, so it is cancelled rather than
            # quietly left resting where it would never be modelled again.
            self.oms.on_cancel(order.client_order_id)
        return fills

    def place_many(
        self, orders: Sequence[Order], books: Mapping[str, OrderBook]
    ) -> list[Fill]:
        out: list[Fill] = []
        for order in orders:
            out.extend(self.place(order, books))
        return out

    # ---- account -------------------------------------------------------
    @property
    def cash(self) -> float:
        assert self.oms is not None
        return self.oms.cash_cents

    def positions(self) -> dict[str, Position]:
        assert self.oms is not None
        return self.oms.positions

    def equity(self, books: Mapping[str, OrderBook]) -> float:
        assert self.oms is not None
        return self.oms.mark_to_market(books)

    def mark(self, books: Mapping[str, OrderBook]) -> None:
        """Refresh the risk engine's view of equity so the kill switch works."""
        assert self.oms is not None
        self.oms.risk.update_equity(self.equity(books))

    def summary(self) -> dict[str, object]:
        assert self.oms is not None
        out = dict(self.oms.summary())
        out["pnl_cents"] = self.oms.cash_cents - self.starting_cash_cents
        return out
