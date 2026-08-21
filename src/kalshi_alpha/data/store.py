"""Columnar tick storage.

Book snapshots are stored **flattened**: one row per (timestamp, ticker) with
the top ``depth`` levels of each side as separate columns. Storing a nested
list per row is more faithful to the wire format but makes every downstream
query pay a deserialisation cost, and the analysis code overwhelmingly wants
columns.

Partitioning is by event ticker and UTC date, which is the access pattern the
research code actually has -- "give me every book for CPI on the release day" --
so a study touches a handful of files rather than scanning the corpus.

Parquet with dictionary-encoded tickers and ZSTD keeps a day of level-2 data
small enough to keep the whole history on a laptop, which matters because the
alternative -- re-pulling from the API -- is rate-limited and not reproducible.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from kalshi_alpha.types import Level, OrderBook, Side, Trade

DEFAULT_DEPTH = 5


def book_to_row(book: OrderBook, depth: int = DEFAULT_DEPTH) -> dict[str, float | str | int]:
    """Flatten one snapshot to a single row."""
    row: dict[str, float | str | int] = {
        "ts": book.ts,
        "ticker": book.ticker,
        "best_yes_bid": book.best_yes_bid if book.best_yes_bid is not None else -1,
        "best_yes_ask": book.best_yes_ask if book.best_yes_ask is not None else -1,
        "mid": book.mid if book.mid is not None else float("nan"),
        "microprice": book.microprice if book.microprice is not None else float("nan"),
    }
    for i in range(depth):
        yb = book.yes_bids[i] if i < len(book.yes_bids) else None
        nb = book.no_bids[i] if i < len(book.no_bids) else None
        row[f"yb_p{i}"] = yb.price if yb else -1
        row[f"yb_s{i}"] = yb.size if yb else 0
        row[f"nb_p{i}"] = nb.price if nb else -1
        row[f"nb_s{i}"] = nb.size if nb else 0
    return row


def row_to_book(row: pd.Series, depth: int = DEFAULT_DEPTH) -> OrderBook:
    """Inverse of :func:`book_to_row`."""
    yes = [
        Level(int(row[f"yb_p{i}"]), int(row[f"yb_s{i}"]))
        for i in range(depth)
        if int(row.get(f"yb_p{i}", -1)) > 0 and int(row.get(f"yb_s{i}", 0)) > 0
    ]
    no = [
        Level(int(row[f"nb_p{i}"]), int(row[f"nb_s{i}"]))
        for i in range(depth)
        if int(row.get(f"nb_p{i}", -1)) > 0 and int(row.get(f"nb_s{i}", 0)) > 0
    ]
    return OrderBook(
        ticker=str(row["ticker"]),
        ts=float(row["ts"]),
        yes_bids=tuple(sorted(yes, key=lambda x: -x.price)),
        no_bids=tuple(sorted(no, key=lambda x: -x.price)),
    )


def books_to_frame(books: Iterable[OrderBook], depth: int = DEFAULT_DEPTH) -> pd.DataFrame:
    rows = [book_to_row(b, depth) for b in books]
    return pd.DataFrame(rows)


def frame_to_books(df: pd.DataFrame, depth: int = DEFAULT_DEPTH) -> list[OrderBook]:
    return [row_to_book(row, depth) for _, row in df.iterrows()]


def trades_to_frame(trades: Iterable[Trade]) -> pd.DataFrame:
    rows = [
        {
            "ts": t.ts,
            "ticker": t.ticker,
            "price": t.price,
            "size": t.size,
            "taker_side": t.taker_side.value,
            "signed_size": t.signed_size,
        }
        for t in trades
    ]
    return pd.DataFrame(rows)


def frame_to_trades(df: pd.DataFrame) -> list[Trade]:
    return [
        Trade(
            ticker=str(r["ticker"]),
            ts=float(r["ts"]),
            price=int(r["price"]),
            size=int(r["size"]),
            taker_side=Side(str(r["taker_side"])),
        )
        for _, r in df.iterrows()
    ]


@dataclass
class TickStore:
    """Parquet-backed store partitioned by event ticker and UTC date."""

    root: Path
    depth: int = DEFAULT_DEPTH
    compression: str = "zstd"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- paths ---------------------------------------------------------
    def _partition(self, kind: str, event_ticker: str, ts: float) -> Path:
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        path = self.root / kind / f"event={event_ticker}" / f"date={day}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---- writes --------------------------------------------------------
    def write_books(
        self, books: Sequence[OrderBook], event_ticker: str, name: str = "part"
    ) -> Path | None:
        if not books:
            return None
        df = books_to_frame(books, self.depth)
        target = self._partition("books", event_ticker, float(books[0].ts)) / f"{name}.parquet"
        df.to_parquet(target, index=False, compression=self.compression)
        return target

    def write_trades(
        self, trades: Sequence[Trade], event_ticker: str, name: str = "part"
    ) -> Path | None:
        if not trades:
            return None
        df = trades_to_frame(trades)
        target = self._partition("trades", event_ticker, float(trades[0].ts)) / f"{name}.parquet"
        df.to_parquet(target, index=False, compression=self.compression)
        return target

    def write_frame(self, df: pd.DataFrame, kind: str, name: str) -> Path:
        path = self.root / kind
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{name}.parquet"
        df.to_parquet(target, index=False, compression=self.compression)
        return target

    # ---- reads ---------------------------------------------------------
    def read(self, kind: str, event_ticker: str | None = None) -> pd.DataFrame:
        base = self.root / kind
        if not base.exists():
            return pd.DataFrame()
        pattern = f"event={event_ticker}/**/*.parquet" if event_ticker else "**/*.parquet"
        files = sorted(base.glob(pattern))
        if not files:
            return pd.DataFrame()
        frames = [pd.read_parquet(f) for f in files]
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values("ts").reset_index(drop=True) if "ts" in out else out

    def read_books(self, event_ticker: str | None = None) -> pd.DataFrame:
        return self.read("books", event_ticker)

    def read_trades(self, event_ticker: str | None = None) -> pd.DataFrame:
        return self.read("trades", event_ticker)

    def events(self, kind: str = "books") -> list[str]:
        base = self.root / kind
        if not base.exists():
            return []
        return sorted(p.name.split("=", 1)[1] for p in base.glob("event=*") if p.is_dir())

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kind in ("books", "trades"):
            base = self.root / kind
            files = list(base.glob("**/*.parquet")) if base.exists() else []
            out[f"{kind}_files"] = len(files)
            out[f"{kind}_bytes"] = sum(f.stat().st_size for f in files)
        return out
