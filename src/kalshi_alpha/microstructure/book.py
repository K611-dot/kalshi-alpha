"""Incremental level-2 book maintenance.

A websocket feed sends one ``orderbook_snapshot`` followed by a stream of
``orderbook_delta`` messages. Reconstructing the book from those deltas is
where most real trading bugs live, so this builder is deliberately strict:

* every message carries a monotonically increasing sequence number, and a gap
  raises :class:`SequenceGap` rather than silently corrupting state;
* a delta that would drive a level negative raises, because that means our
  state has already diverged from the exchange;
* the builder is pure -- it holds no I/O -- so the exact same code path is
  exercised by the live feed, the replay harness and the unit tests.

Silently repairing a bad book is worse than stopping: you keep quoting against
a book that no longer exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from kalshi_alpha.types import Level, OrderBook, Side


class SequenceGap(RuntimeError):
    """Raised when a delta arrives out of order, implying dropped messages."""


class BookCorruption(RuntimeError):
    """Raised when a delta would produce an impossible book state."""


def apply_delta(
    levels: Mapping[int, int], price: int, delta: int, allow_negative: bool = False
) -> dict[int, int]:
    """Apply a size delta at one price, returning a new price->size mapping."""
    out = dict(levels)
    new_size = out.get(price, 0) + delta
    if new_size < 0 and not allow_negative:
        raise BookCorruption(
            f"delta {delta:+d} at {price}c would drive size to {new_size}; local book diverged"
        )
    if new_size <= 0:
        out.pop(price, None)
    else:
        out[price] = new_size
    return out


@dataclass
class BookBuilder:
    """Maintains one market's book from a snapshot plus deltas."""

    ticker: str
    yes: dict[int, int] = field(default_factory=dict)
    no: dict[int, int] = field(default_factory=dict)
    seq: int = 0
    ts: float = 0.0
    strict_sequence: bool = True
    updates: int = 0
    resyncs: int = 0

    # ---- state transitions --------------------------------------------
    def snapshot(
        self,
        yes_levels: Iterable[tuple[int, int]],
        no_levels: Iterable[tuple[int, int]],
        seq: int,
        ts: float,
    ) -> None:
        """Replace local state wholesale. Also used to recover from a gap."""
        self.yes = {int(p): int(s) for p, s in yes_levels if s > 0}
        self.no = {int(p): int(s) for p, s in no_levels if s > 0}
        if self.seq:
            self.resyncs += 1
        self.seq = int(seq)
        self.ts = float(ts)

    def delta(self, side: Side, price: int, change: int, seq: int, ts: float) -> None:
        """Apply one incremental update."""
        if self.strict_sequence and seq != self.seq + 1:
            raise SequenceGap(
                f"{self.ticker}: expected seq {self.seq + 1}, received {seq} "
                f"({seq - self.seq - 1} messages missing)"
            )
        if side is Side.YES:
            self.yes = apply_delta(self.yes, int(price), int(change))
        else:
            self.no = apply_delta(self.no, int(price), int(change))
        self.seq = int(seq)
        self.ts = float(ts)
        self.updates += 1

    # ---- output --------------------------------------------------------
    def book(self) -> OrderBook:
        return OrderBook(
            ticker=self.ticker,
            ts=self.ts,
            yes_bids=tuple(
                sorted((Level(p, s) for p, s in self.yes.items()), key=lambda x: -x.price)
            ),
            no_bids=tuple(
                sorted((Level(p, s) for p, s in self.no.items()), key=lambda x: -x.price)
            ),
        )

    @property
    def healthy(self) -> bool:
        book = self.book()
        return book.is_two_sided and not book.is_crossed


class BookSet:
    """A keyed collection of :class:`BookBuilder` objects for a whole event."""

    def __init__(self, strict_sequence: bool = True) -> None:
        self._builders: dict[str, BookBuilder] = {}
        self._strict = strict_sequence

    def builder(self, ticker: str) -> BookBuilder:
        if ticker not in self._builders:
            self._builders[ticker] = BookBuilder(ticker, strict_sequence=self._strict)
        return self._builders[ticker]

    def snapshot(
        self,
        ticker: str,
        yes_levels: Iterable[tuple[int, int]],
        no_levels: Iterable[tuple[int, int]],
        seq: int,
        ts: float,
    ) -> None:
        self.builder(ticker).snapshot(yes_levels, no_levels, seq, ts)

    def delta(self, ticker: str, side: Side, price: int, change: int, seq: int, ts: float) -> None:
        self.builder(ticker).delta(side, price, change, seq, ts)

    def books(self) -> dict[str, OrderBook]:
        return {tk: b.book() for tk, b in self._builders.items()}

    def health(self) -> dict[str, int]:
        return {
            "markets": len(self._builders),
            "updates": sum(b.updates for b in self._builders.values()),
            "resyncs": sum(b.resyncs for b in self._builders.values()),
            "unhealthy": sum(not b.healthy for b in self._builders.values()),
        }

    def __len__(self) -> int:
        return len(self._builders)

    def __contains__(self, ticker: object) -> bool:
        return ticker in self._builders
