"""Core domain types for binary event contracts.

Design notes
------------
Kalshi quotes everything in **integer cents** on a 1..99 grid, and the exchange
publishes *two bid books* per market -- a YES bid book and a NO bid book. There
is no separately published ask book, because on a binary contract::

    buy  YES @ p   ==  sell NO  @ (100 - p)
    sell YES @ p   ==  buy  NO  @ (100 - p)

so the YES ask ladder is the mirror image of the NO bid ladder. We model this
faithfully rather than flattening to a synthetic bid/ask, because the mirror
relation is exactly what makes naive "buy YES + buy NO < $1" arbitrage
*impossible inside a single market* (see docs/METHODOLOGY.md) and forces the
real edge to live across markets, across strike ladders, and across venues.

All prices are integers in cents to avoid floating-point drift in P&L
accounting; probabilities are floats in [0, 1] only at the modelling boundary.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

Cents = int

MIN_PRICE: Cents = 1
MAX_PRICE: Cents = 99
PAYOUT: Cents = 100


class Side(str, Enum):
    """Which leg of the binary contract is being held."""

    YES = "yes"
    NO = "no"

    @property
    def other(self) -> Side:
        return Side.NO if self is Side.YES else Side.YES


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Action.BUY else -1


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(str, Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(str, Enum):
    NEW = "new"
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Level:
    """One price level of a bid book."""

    price: Cents
    size: int

    def __post_init__(self) -> None:
        if not 0 < self.price < PAYOUT:
            raise ValueError(f"price must be in (0, 100) cents, got {self.price}")
        if self.size < 0:
            raise ValueError(f"size must be non-negative, got {self.size}")


@dataclass(frozen=True, slots=True)
class OrderBook:
    """Level-2 snapshot of a single binary market.

    ``yes_bids`` and ``no_bids`` are both *bid* books sorted by descending price,
    exactly as the exchange publishes them.
    """

    ticker: str
    ts: float
    yes_bids: tuple[Level, ...] = ()
    no_bids: tuple[Level, ...] = ()

    # ---- construction -------------------------------------------------
    @classmethod
    def from_levels(
        cls,
        ticker: str,
        ts: float,
        yes_bids: Iterable[tuple[int, int]] = (),
        no_bids: Iterable[tuple[int, int]] = (),
    ) -> OrderBook:
        yb = tuple(sorted((Level(p, s) for p, s in yes_bids if s > 0), key=lambda x: -x.price))
        nb = tuple(sorted((Level(p, s) for p, s in no_bids if s > 0), key=lambda x: -x.price))
        return cls(ticker=ticker, ts=ts, yes_bids=yb, no_bids=nb)

    # ---- derived ladders ----------------------------------------------
    @property
    def yes_asks(self) -> tuple[Level, ...]:
        """YES offer ladder, ascending price -- the mirror of the NO bid book."""
        return tuple(
            sorted(
                (Level(PAYOUT - lv.price, lv.size) for lv in self.no_bids),
                key=lambda x: x.price,
            )
        )

    @property
    def no_asks(self) -> tuple[Level, ...]:
        return tuple(
            sorted(
                (Level(PAYOUT - lv.price, lv.size) for lv in self.yes_bids),
                key=lambda x: x.price,
            )
        )

    def ladder(self, side: Side, action: Action) -> tuple[Level, ...]:
        """The ladder you consume when taking liquidity for (side, action)."""
        if side is Side.YES:
            return self.yes_asks if action is Action.BUY else self.yes_bids
        return self.no_asks if action is Action.BUY else self.no_bids

    # ---- top of book ---------------------------------------------------
    @property
    def best_yes_bid(self) -> Cents | None:
        return self.yes_bids[0].price if self.yes_bids else None

    @property
    def best_no_bid(self) -> Cents | None:
        return self.no_bids[0].price if self.no_bids else None

    @property
    def best_yes_ask(self) -> Cents | None:
        b = self.best_no_bid
        return None if b is None else PAYOUT - b

    @property
    def best_yes_bid_size(self) -> int:
        return self.yes_bids[0].size if self.yes_bids else 0

    @property
    def best_yes_ask_size(self) -> int:
        return self.no_bids[0].size if self.no_bids else 0

    @property
    def is_crossed(self) -> bool:
        b, a = self.best_yes_bid, self.best_yes_ask
        return b is not None and a is not None and b >= a

    @property
    def is_two_sided(self) -> bool:
        return bool(self.yes_bids) and bool(self.no_bids)

    # ---- summary statistics -------------------------------------------
    @property
    def spread(self) -> Cents | None:
        b, a = self.best_yes_bid, self.best_yes_ask
        return None if b is None or a is None else a - b

    @property
    def mid(self) -> float | None:
        """Naive mid in cents."""
        b, a = self.best_yes_bid, self.best_yes_ask
        if b is None:
            return None if a is None else float(a)
        if a is None:
            return float(b)
        return (b + a) / 2.0

    @property
    def microprice(self) -> float | None:
        """Size-weighted mid; bid/ask weights are *crossed* on purpose.

        Heavier size resting on the bid means the bid is more likely to be
        consumed before the ask moves, which pushes fair value toward the ask
        (Stoikov, 2018).
        """
        b, a = self.best_yes_bid, self.best_yes_ask
        if b is None or a is None:
            return self.mid
        qb, qa = self.best_yes_bid_size, self.best_yes_ask_size
        if qb + qa == 0:
            return self.mid
        return (qb * a + qa * b) / (qb + qa)

    @property
    def imbalance(self) -> float | None:
        """Top-of-book order imbalance in [-1, 1]; +1 = all bid."""
        qb, qa = self.best_yes_bid_size, self.best_yes_ask_size
        if qb + qa == 0:
            return None
        return (qb - qa) / (qb + qa)

    def depth_imbalance(self, levels: int = 5) -> float | None:
        qb = sum(lv.size for lv in self.yes_bids[:levels])
        qa = sum(lv.size for lv in self.yes_asks[:levels])
        if qb + qa == 0:
            return None
        return (qb - qa) / (qb + qa)

    def notional_depth(self, levels: int = 5) -> int:
        """Total cents of resting notional across both sides (risk-capacity proxy)."""
        bid = sum(lv.price * lv.size for lv in self.yes_bids[:levels])
        ask = sum(lv.price * lv.size for lv in self.yes_asks[:levels])
        return bid + ask

    # ---- liquidity consumption ----------------------------------------
    def walk(self, side: Side, action: Action, qty: int) -> tuple[int, Cents]:
        """Walk the book for ``qty`` contracts.

        Returns ``(filled, cash_cents)`` where ``cash_cents`` is the absolute
        amount paid (BUY) or received (SELL), excluding fees.
        """
        if qty <= 0:
            return 0, 0
        remaining, cash = qty, 0
        for lv in self.ladder(side, action):
            if remaining == 0:
                break
            take = min(remaining, lv.size)
            cash += take * lv.price
            remaining -= take
        return qty - remaining, cash

    def vwap(self, side: Side, action: Action, qty: int) -> float | None:
        filled, cash = self.walk(side, action, qty)
        return None if filled == 0 else cash / filled

    def sweep_cost(self, side: Side, action: Action, qty: int) -> Cents | None:
        """Cash cost to fully execute ``qty``; ``None`` if the book is too thin."""
        filled, cash = self.walk(side, action, qty)
        return cash if filled == qty else None

    def available(self, side: Side, action: Action, max_levels: int | None = None) -> int:
        lad = self.ladder(side, action)
        if max_levels is not None:
            lad = lad[:max_levels]
        return sum(lv.size for lv in lad)

    def implied_prob(self, use: str = "mid") -> float | None:
        """Risk-neutral probability implied by the book, in [0, 1]."""
        px = self.microprice if use == "micro" else self.mid
        return None if px is None else px / PAYOUT


@dataclass(frozen=True, slots=True)
class Trade:
    ticker: str
    ts: float
    price: Cents  # YES price
    size: int
    taker_side: Side  # side the aggressor bought

    @property
    def signed_size(self) -> int:
        """+size if the aggressor lifted the YES offer, -size if they hit the bid."""
        return self.size if self.taker_side is Side.YES else -self.size


@dataclass(frozen=True, slots=True)
class MarketMeta:
    """Static description of a market, including the logical claim it settles on."""

    ticker: str
    event_ticker: str
    title: str = ""
    strike: float | None = None
    # "gte"/"gt"/"lte"/"lt" for threshold markets, "between" for buckets, None otherwise
    strike_type: str | None = None
    strike_upper: float | None = None
    close_ts: float | None = None
    category: str = ""

    def payoff(self, outcome: float) -> int:
        """Realised YES payoff (0 or 1) given the settlement value of the underlying."""
        if self.strike_type is None or self.strike is None:
            raise ValueError(f"{self.ticker} has no parseable strike; cannot evaluate payoff")
        if self.strike_type == "gte":
            return int(outcome >= self.strike)
        if self.strike_type == "gt":
            return int(outcome > self.strike)
        if self.strike_type == "lte":
            return int(outcome <= self.strike)
        if self.strike_type == "lt":
            return int(outcome < self.strike)
        if self.strike_type == "between":
            hi = math.inf if self.strike_upper is None else self.strike_upper
            return int(self.strike <= outcome < hi)
        raise ValueError(f"unknown strike_type {self.strike_type!r}")


@dataclass(frozen=True, slots=True)
class EventGroup:
    """A set of markets whose YES outcomes are mutually exclusive and exhaustive.

    Exactly one member settles YES, so fair YES prices must sum to $1.00.
    Kalshi's "which candidate", "which Fed decision", and bucketed-range events
    are all of this form.
    """

    event_ticker: str
    tickers: tuple[str, ...]
    exhaustive: bool = True
    label: str = ""

    def __len__(self) -> int:
        return len(self.tickers)


@dataclass(frozen=True, slots=True)
class LadderGroup:
    """A monotone strike ladder, e.g. 'CPI >= 2.9%', 'CPI >= 3.0%', ...

    For a ``gte``/``gt`` ladder, P(X >= k) is non-increasing in k. Any violation
    of that ordering in the *executable* prices is a model-free arbitrage.
    """

    event_ticker: str
    tickers: tuple[str, ...]  # ordered by increasing strike
    strikes: tuple[float, ...]
    direction: str = "gte"  # "gte"/"gt" => decreasing; "lte"/"lt" => increasing
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.tickers) != len(self.strikes):
            raise ValueError("tickers and strikes must be the same length")
        if list(self.strikes) != sorted(self.strikes):
            raise ValueError("strikes must be supplied in increasing order")

    @property
    def decreasing(self) -> bool:
        return self.direction in ("gte", "gt")


@dataclass(frozen=True, slots=True)
class Order:
    ticker: str
    side: Side
    action: Action
    qty: int
    price: Cents | None = None  # None => market order
    order_type: OrderType = OrderType.LIMIT
    tif: TimeInForce = TimeInForce.GTC
    client_order_id: str = ""
    ts: float = 0.0
    tag: str = ""

    @property
    def is_marketable(self) -> bool:
        return self.order_type is OrderType.MARKET or self.price is None


@dataclass(frozen=True, slots=True)
class Fill:
    ticker: str
    ts: float
    side: Side
    action: Action
    qty: int
    price: Cents
    fee_cents: Cents = 0
    order_id: str = ""
    liquidity: str = "taker"  # "taker" | "maker"
    tag: str = ""

    @property
    def cash_delta(self) -> int:
        """Signed change to the cash balance in cents (negative = cash out)."""
        gross = self.qty * self.price
        return (-gross if self.action is Action.BUY else gross) - self.fee_cents


@dataclass
class Position:
    """Net position in a single market.

    Exposure is tracked in YES-equivalent contracts, because a NO contract is
    economically a short YES. That netting is correct for *directional* risk but
    it silently drops a constant, and the constant is worth real money::

        terminal = 100 * s * Y + 100 * (1 - s) * N
                 = 100 * s * (Y - N) + 100 * N

    The first term is the netted YES exposure; ``100 * N`` is a **guaranteed**
    payment that a long NO position collects no matter how the event resolves.
    Holding one YES and one NO in the same market nets to zero exposure and
    still pays exactly $1 -- which is precisely why the crossed-book and
    ladder arbitrages are riskless in the first place.

    So the unconditional leg is accumulated separately in ``no_credit_cents``.
    Netting exposure without it understates P&L by $1 per offsetting pair and
    makes every hedged structure look like a total loss.
    """

    ticker: str
    yes_qty: int = 0
    cash_cents: int = 0  # cumulative signed cash from all fills (fees included)
    no_credit_cents: int = 0  # guaranteed settlement cash from long NO contracts
    fees_cents: int = 0
    fills: list[Fill] = field(default_factory=list)

    def apply(self, fill: Fill) -> None:
        signed = fill.qty * fill.action.sign
        if fill.side is Side.YES:
            self.yes_qty += signed
        else:
            self.yes_qty -= signed
            self.no_credit_cents += signed * PAYOUT
        self.cash_cents += fill.cash_delta
        self.fees_cents += fill.fee_cents
        self.fills.append(fill)

    def mark_to_market(self, yes_price: float) -> float:
        """Total P&L in cents if the position were liquidated at ``yes_price``."""
        return self.cash_cents + self.yes_qty * yes_price + self.no_credit_cents

    def settle(self, yes_outcome: int) -> float:
        """Realised P&L in cents once the market settles YES (1) or NO (0)."""
        return self.cash_cents + self.yes_qty * yes_outcome * PAYOUT + self.no_credit_cents

    @property
    def is_flat(self) -> bool:
        return self.yes_qty == 0


@dataclass(frozen=True, slots=True)
class Leg:
    """One executable leg of a multi-market arbitrage."""

    ticker: str
    side: Side
    action: Action
    qty: int
    price: Cents
    fee_cents: Cents = 0

    @property
    def cash_out(self) -> int:
        """Cents paid (positive) or received (negative), fees included."""
        gross = self.qty * self.price
        return (gross if self.action is Action.BUY else -gross) + self.fee_cents

    @property
    def yes_delta(self) -> int:
        """Signed YES-equivalent exposure created by this leg."""
        signed = self.qty * self.action.sign
        return signed if self.side is Side.YES else -signed

    def describe(self) -> str:
        return (
            f"{self.action.value.upper():4s} {self.qty:>4d} {self.ticker} "
            f"{self.side.value.upper():3s} @ {self.price:>2d}c"
        )


@dataclass(frozen=True, slots=True)
class ArbOpportunity:
    """A detected, fee-inclusive, size-constrained arbitrage."""

    kind: str
    event_ticker: str
    legs: tuple[Leg, ...]
    cost_cents: int  # net cash out to establish (may be negative)
    worst_case_pnl_cents: int  # guaranteed profit across all settlement states
    best_case_pnl_cents: int
    ts: float = 0.0
    detail: str = ""

    @property
    def qty(self) -> int:
        return max((leg.qty for leg in self.legs), default=0)

    @property
    def capital_at_risk_cents(self) -> int:
        return max(self.cost_cents, 0)

    @property
    def return_on_capital(self) -> float:
        cap = self.capital_at_risk_cents
        return math.inf if cap == 0 else self.worst_case_pnl_cents / cap

    def describe(self) -> str:
        roc = self.return_on_capital
        roc_s = "inf" if math.isinf(roc) else f"{roc:+.2%}"
        head = (
            f"[{self.kind}] {self.event_ticker}  "
            f"cost={self.cost_cents}c  guaranteed=+{self.worst_case_pnl_cents}c  roc={roc_s}"
        )
        return "\n".join([head, *(f"    {lg.describe()}" for lg in self.legs)])


def clamp_price(p: float) -> Cents:
    """Snap a continuous price to the tradeable 1..99 cent grid."""
    return int(min(MAX_PRICE, max(MIN_PRICE, round(p))))


def prob_to_cents(p: float) -> Cents:
    return clamp_price(p * PAYOUT)


def summarize_books(books: Sequence[OrderBook]) -> dict[str, float]:
    """Cheap health summary used in logs and reports."""
    mids = [b.mid for b in books if b.mid is not None]
    spreads = [b.spread for b in books if b.spread is not None]
    return {
        "n_books": float(len(books)),
        "n_two_sided": float(sum(b.is_two_sided for b in books)),
        "n_crossed": float(sum(b.is_crossed for b in books)),
        "mean_mid": float(sum(mids) / len(mids)) if mids else float("nan"),
        "mean_spread": float(sum(spreads) / len(spreads)) if spreads else float("nan"),
    }
