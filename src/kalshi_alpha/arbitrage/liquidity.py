"""Book-walking with exact per-level fees.

The exchange charges its taker fee **per fill**, so an order that sweeps three
price levels pays three separately-rounded fees. Sizing an arbitrage off the
top-of-book price alone systematically overstates the edge; every quantity
decision in this package goes through :func:`sweep`, which walks the real
ladder and accumulates the real fee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kalshi_alpha.arbitrage.fees import DEFAULT_TAKER_RATE, exact_taker_fee_cents
from kalshi_alpha.types import Action, Cents, OrderBook, Side


@dataclass(frozen=True, slots=True)
class Sweep:
    """Result of consuming ``requested`` contracts from one side of a book."""

    requested: int
    filled: int
    cash_cents: Cents  # absolute gross cash (paid on BUY, received on SELL)
    fee_cents: Cents
    levels: tuple[tuple[Cents, int], ...] = ()

    @property
    def complete(self) -> bool:
        return self.filled == self.requested and self.requested > 0

    @property
    def vwap(self) -> float:
        return math.nan if self.filled == 0 else self.cash_cents / self.filled

    @property
    def worst_price(self) -> Cents | None:
        return self.levels[-1][0] if self.levels else None

    def cost_out(self, action: Action) -> int:
        """Signed cash out (positive = money leaves the account), fees included."""
        gross = self.cash_cents if action is Action.BUY else -self.cash_cents
        return gross + self.fee_cents


def haircut_size(size: int, haircut: float) -> int:
    """Discount displayed size to allow for stale or phantom liquidity."""
    if haircut >= 1.0:
        return size
    return int(math.floor(size * max(0.0, haircut)))


def sweep(
    book: OrderBook,
    side: Side,
    action: Action,
    qty: int,
    taker_rate: float = DEFAULT_TAKER_RATE,
    size_haircut: float = 1.0,
    max_levels: int | None = None,
) -> Sweep:
    """Walk ``qty`` contracts through ``book`` and price them with exact fees."""
    if qty <= 0:
        return Sweep(requested=max(qty, 0), filled=0, cash_cents=0, fee_cents=0)

    ladder = book.ladder(side, action)
    if max_levels is not None:
        ladder = ladder[:max_levels]

    remaining = qty
    cash = 0
    fee = 0
    used: list[tuple[int, int]] = []
    for lv in ladder:
        if remaining == 0:
            break
        avail = haircut_size(lv.size, size_haircut)
        if avail <= 0:
            continue
        take = min(remaining, avail)
        cash += take * lv.price
        fee += exact_taker_fee_cents(lv.price, take, taker_rate)
        used.append((lv.price, take))
        remaining -= take

    return Sweep(
        requested=qty,
        filled=qty - remaining,
        cash_cents=cash,
        fee_cents=fee,
        levels=tuple(used),
    )


def max_executable(
    book: OrderBook,
    side: Side,
    action: Action,
    size_haircut: float = 1.0,
    max_levels: int | None = None,
) -> int:
    """Total contracts obtainable from one side of the book after haircut."""
    ladder = book.ladder(side, action)
    if max_levels is not None:
        ladder = ladder[:max_levels]
    return sum(haircut_size(lv.size, size_haircut) for lv in ladder)


def marginal_cost_curve(
    book: OrderBook,
    side: Side,
    action: Action,
    max_qty: int,
    taker_rate: float = DEFAULT_TAKER_RATE,
    size_haircut: float = 1.0,
) -> list[int]:
    """Total all-in cost for 1..max_qty contracts.

    The curve is convex by construction (levels are consumed best-first), which
    is what lets the sizing search stop at the first quantity where marginal
    profit turns negative.
    """
    out: list[int] = []
    for q in range(1, max_qty + 1):
        sw = sweep(book, side, action, q, taker_rate, size_haircut)
        if not sw.complete:
            break
        out.append(sw.cash_cents + sw.fee_cents)
    return out
