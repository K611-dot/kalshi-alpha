"""Fill simulation.

Most backtests assume you get filled at the mid, or at the touch, instantly.
That single assumption is responsible for more phantom alpha than any modelling
error, because it hands the strategy the spread it would actually have paid.

This model charges for everything a real order pays:

* **Latency.** An order submitted at ``t`` reaches the matching engine at
  ``t + order_latency``, and it trades against the book *then*, not the book
  the signal was computed on. Cancels are slow too, which is what makes resting
  orders vulnerable.
* **Queue position.** A passive order joins the back of the queue at its price.
  It only fills after the size that was already resting there is consumed --
  the single most important detail for any strategy that quotes, and the one
  most often skipped.
* **Queue leakage.** Some of the queue ahead cancels rather than trades, which
  advances us faster than raw volume implies. ``queue_leakage`` is the fraction
  that disappears without printing.
* **Adverse selection.** Aggressive orders are sometimes rejected because the
  level moved before arrival; passive orders fill preferentially when the price
  is about to move through them. Getting filled is not free information.
* **Depth.** Marketable orders walk the ladder and pay the real VWAP, with the
  per-level fee applied to each fill separately.

The resulting fills are pessimistic by design. A strategy that survives this
model has a chance; one that only works with mid fills does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kalshi_alpha.arbitrage.fees import exact_taker_fee_cents, maker_fee_cents
from kalshi_alpha.config import ExecutionConfig, FeeConfig
from kalshi_alpha.types import (
    PAYOUT,
    Action,
    Fill,
    Order,
    OrderBook,
    OrderStatus,
    Side,
    Trade,
)


@dataclass
class RestingOrder:
    """A passive order waiting in the queue at its price level."""

    order: Order
    submitted_ts: float
    live_ts: float  # when it actually reaches the book
    queue_ahead: float
    filled: int = 0
    status: OrderStatus = OrderStatus.PENDING
    cancel_requested_ts: float | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.order.qty - self.filled)

    @property
    def is_live(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    def matches_trade(self, trade: Trade) -> bool:
        """Would this print have executed against our resting order?

        A resting BUY YES at price ``p`` sits on the YES bid. It trades when an
        aggressor *sells* YES at ``p`` -- which the tape reports as a trade at
        YES price ``p`` whose taker took the NO side. The mirror holds for a
        resting BUY NO at ``q``: the print shows YES price ``100 - q`` with a
        YES-side taker.
        """
        o = self.order
        if o.price is None or trade.ticker != o.ticker:
            return False
        if o.side is Side.YES and o.action is Action.BUY:
            return trade.price == o.price and trade.taker_side is Side.NO
        if o.side is Side.NO and o.action is Action.BUY:
            return trade.price == PAYOUT - o.price and trade.taker_side is Side.YES
        if o.side is Side.YES and o.action is Action.SELL:
            return trade.price == o.price and trade.taker_side is Side.YES
        return trade.price == PAYOUT - o.price and trade.taker_side is Side.NO


@dataclass
class FillModel:
    """Turns orders plus market data into fills."""

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    seed: int = 0
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # ---- fees ----------------------------------------------------------
    def fee(self, price: int, qty: int, liquidity: str) -> int:
        if liquidity == "maker":
            return (
                maker_fee_cents(price, qty, self.fees.maker_rate)
                if self.fees.maker_fees_enabled
                else 0
            )
        return exact_taker_fee_cents(price, qty, self.fees.taker_rate)

    # ---- aggressive ----------------------------------------------------
    def fill_aggressive(self, order: Order, book: OrderBook, ts: float) -> list[Fill]:
        """Walk the book for a marketable order, one fill per price level.

        A limit price still applies: levels worse than the limit are not taken,
        so a marketable-limit order can partially fill and stop -- which is what
        actually happens when you cross into a book that is thinner than the
        signal assumed.
        """
        if self._rng.random() < self.execution.adverse_reject_prob:
            return []  # the level moved before we arrived

        ladder = book.ladder(order.side, order.action)
        remaining = order.qty
        out: list[Fill] = []
        for level in ladder:
            if remaining <= 0:
                break
            if order.price is not None and order.order_type.value == "limit":
                if order.action is Action.BUY and level.price > order.price:
                    break
                if order.action is Action.SELL and level.price < order.price:
                    break
            take = min(remaining, level.size)
            if take <= 0:
                continue
            out.append(
                Fill(
                    ticker=order.ticker,
                    ts=ts,
                    side=order.side,
                    action=order.action,
                    qty=take,
                    price=level.price,
                    fee_cents=self.fee(level.price, take, "taker"),
                    order_id=order.client_order_id,
                    liquidity="taker",
                    tag=order.tag,
                )
            )
            remaining -= take
        return out

    # ---- passive -------------------------------------------------------
    def initial_queue(self, order: Order, book: OrderBook) -> float:
        """Size already resting at our price when the order arrives.

        We join the back of it. Orders that *improve* the book start with an
        empty queue, which is the compensation for showing a better price.
        """
        if order.price is None:
            return 0.0
        # Every resting order lives on one of the two bid books. Selling YES at
        # p is the same resting interest as buying NO at 100 - p, so normalise
        # to (book, price) before counting the queue.
        if order.action is Action.BUY:
            levels = book.yes_bids if order.side is Side.YES else book.no_bids
            price = order.price
        else:
            levels = book.no_bids if order.side is Side.YES else book.yes_bids
            price = PAYOUT - order.price
        return float(sum(lv.size for lv in levels if lv.price == price))

    def apply_trade(self, resting: RestingOrder, trade: Trade) -> list[Fill]:
        """Consume the queue with one print and fill whatever reaches us."""
        if not resting.is_live or not resting.matches_trade(trade):
            return []

        # Part of the queue ahead cancels instead of trading, so each printed
        # contract advances us by more than one place.
        leak = float(np.clip(self.execution.queue_leakage, 0.0, 0.95))
        advance = trade.size / (1.0 - leak)

        if resting.queue_ahead >= advance:
            resting.queue_ahead -= advance
            return []

        available = int(np.floor(advance - resting.queue_ahead))
        resting.queue_ahead = 0.0
        take = min(resting.remaining, max(available, 0))
        if take <= 0:
            return []

        resting.filled += take
        resting.status = (
            OrderStatus.FILLED if resting.remaining == 0 else OrderStatus.PARTIALLY_FILLED
        )
        price = resting.order.price
        assert price is not None
        return [
            Fill(
                ticker=resting.order.ticker,
                ts=trade.ts,
                side=resting.order.side,
                action=resting.order.action,
                qty=take,
                price=price,
                fee_cents=self.fee(price, take, "maker"),
                order_id=resting.order.client_order_id,
                liquidity="maker",
                tag=resting.order.tag,
            )
        ]

    def sweep_through(self, resting: RestingOrder, book: OrderBook, ts: float) -> list[Fill]:
        """Fill a resting order the market has traded straight past.

        If the book's touch moves through our price without us having been
        filled, we were adversely selected: the market decided our price was
        wrong and took the level out. Modelling this is what stops a quoting
        strategy from looking risk-free.
        """
        o = resting.order
        if not resting.is_live or o.price is None:
            return []
        best_bid, best_ask = book.best_yes_bid, book.best_yes_ask
        if best_bid is None or best_ask is None:
            return []

        yes_price = o.price if o.side is Side.YES else PAYOUT - o.price
        buying_yes = (o.side is Side.YES) == (o.action is Action.BUY)
        through = (buying_yes and best_ask < yes_price) or (
            not buying_yes and best_bid > yes_price
        )
        if not through:
            return []

        take = resting.remaining
        resting.filled += take
        resting.status = OrderStatus.FILLED
        return [
            Fill(
                ticker=o.ticker,
                ts=ts,
                side=o.side,
                action=o.action,
                qty=take,
                price=o.price,
                fee_cents=self.fee(o.price, take, "maker"),
                order_id=o.client_order_id,
                liquidity="maker",
                tag=o.tag + "|adverse",
            )
        ]
