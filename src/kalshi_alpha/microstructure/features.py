"""Microstructure features.

Every feature here is computable from a book snapshot alone, or from a pair of
consecutive snapshots, so the same code runs online in the execution loop and
offline over a replay without a second implementation to keep in sync.

The headline feature is **order-flow imbalance** (Cont, Kukanov and Stoikov,
2014). Unlike raw trade-sign volume it counts *all* book activity -- new
liquidity posted, liquidity pulled, and liquidity consumed -- and on equity
data it explains substantially more short-horizon price variation than trade
imbalance does. On a prediction market it plays the same role: it is the
cleanest observable proxy for the arrival of information that has not yet been
printed as a trade.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from kalshi_alpha.types import PAYOUT, OrderBook, Trade


def book_features(book: OrderBook, levels: int = 5) -> dict[str, float]:
    """Point-in-time features for a single snapshot."""
    bid, ask = book.best_yes_bid, book.best_yes_ask
    mid = book.mid
    micro = book.microprice
    imbalance = book.imbalance
    depth_imb = book.depth_imbalance(levels)
    feats: dict[str, float] = {
        "ts": book.ts,
        "bid": float(bid) if bid is not None else np.nan,
        "ask": float(ask) if ask is not None else np.nan,
        "mid": float(mid) if mid is not None else np.nan,
        "microprice": float(micro) if micro is not None else np.nan,
        "spread": float(book.spread) if book.spread is not None else np.nan,
        "bid_size": float(book.best_yes_bid_size),
        "ask_size": float(book.best_yes_ask_size),
        "imbalance": imbalance if imbalance is not None else np.nan,
        "depth_imbalance": depth_imb if depth_imb is not None else np.nan,
        "notional_depth": float(book.notional_depth(levels)),
        "crossed": float(book.is_crossed),
    }
    # Micro-mid drift: how far the size-weighted price sits from the naive mid,
    # normalised by the spread. Values near +/-0.5 mean the next print is very
    # likely to happen on that side.
    if mid is not None and micro is not None and book.spread:
        feats["micro_tilt"] = (micro - mid) / book.spread
    else:
        feats["micro_tilt"] = np.nan
    return feats


def order_flow_imbalance(prev: OrderBook, curr: OrderBook) -> float:
    """Cont-Kukanov-Stoikov order-flow imbalance between two snapshots.

    .. math::

        e_n = \\mathbb{1}_{P^b_n \\ge P^b_{n-1}} q^b_n
            - \\mathbb{1}_{P^b_n \\le P^b_{n-1}} q^b_{n-1}
            - \\mathbb{1}_{P^a_n \\le P^a_{n-1}} q^a_n
            + \\mathbb{1}_{P^a_n \\ge P^a_{n-1}} q^a_{n-1}

    Positive values mean net buying pressure: the bid improved or thickened, or
    the offer was consumed or pulled.
    """
    pb0, pb1 = prev.best_yes_bid, curr.best_yes_bid
    pa0, pa1 = prev.best_yes_ask, curr.best_yes_ask
    if pb0 is None or pb1 is None or pa0 is None or pa1 is None:
        return np.nan
    qb0, qb1 = prev.best_yes_bid_size, curr.best_yes_bid_size
    qa0, qa1 = prev.best_yes_ask_size, curr.best_yes_ask_size

    e = 0.0
    if pb1 >= pb0:
        e += qb1
    if pb1 <= pb0:
        e -= qb0
    if pa1 <= pa0:
        e -= qa1
    if pa1 >= pa0:
        e += qa0
    return float(e)


def feature_frame(
    books: Sequence[OrderBook],
    levels: int = 5,
    ofi_window: int = 20,
) -> pd.DataFrame:
    """Build the full feature matrix from a book replay."""
    if not books:
        return pd.DataFrame()

    rows = [book_features(b, levels) for b in books]
    df = pd.DataFrame(rows)

    ofi = [np.nan]
    ofi.extend(order_flow_imbalance(books[i - 1], books[i]) for i in range(1, len(books)))
    df["ofi"] = ofi
    df["ofi_roll"] = df["ofi"].rolling(ofi_window, min_periods=1).sum()

    df["ret"] = df["mid"].diff()
    df["micro_ret"] = df["microprice"].diff()
    # Realised volatility of the mid in cents, annualisation-free: on a
    # prediction market the natural clock is the event, not the year.
    df["rv"] = df["ret"].rolling(ofi_window, min_periods=2).std()
    df["abs_ret"] = df["ret"].abs()
    df["prob"] = df["mid"] / PAYOUT
    # Bernoulli variance of the implied probability: uncertainty is maximal at
    # 50c and collapses at the tails, so price moves near even money are far
    # less informative per cent than moves in the wings.
    df["prob_var"] = df["prob"] * (1.0 - df["prob"])
    return df


def trade_frame(trades: Sequence[Trade]) -> pd.DataFrame:
    """Tabulate a trade tape with signed volume."""
    if not trades:
        return pd.DataFrame(columns=["ts", "price", "size", "signed_size", "notional"])
    df = pd.DataFrame(
        {
            "ts": [t.ts for t in trades],
            "price": [float(t.price) for t in trades],
            "size": [t.size for t in trades],
            "signed_size": [t.signed_size for t in trades],
        }
    )
    df["notional"] = df["price"] * df["size"]
    df["signed_notional"] = df["price"] * df["signed_size"]
    return df


def merge_books_trades(book_df: pd.DataFrame, trade_df: pd.DataFrame) -> pd.DataFrame:
    """As-of join the tape onto the book series (backward, no look-ahead)."""
    if book_df.empty or trade_df.empty:
        out = book_df.copy()
        out["trade_price"] = np.nan
        out["trade_signed_size"] = 0.0
        return out
    left = book_df.sort_values("ts")
    right = trade_df.sort_values("ts").rename(
        columns={"price": "trade_price", "signed_size": "trade_signed_size"}
    )
    return pd.merge_asof(
        left,
        right[["ts", "trade_price", "trade_signed_size"]],
        on="ts",
        direction="backward",
    )


def quote_slippage(book: OrderBook, qty: int, side_yes: bool = True) -> float:
    """Cents of slippage versus the mid when sweeping ``qty`` contracts."""
    from kalshi_alpha.types import Action, Side

    mid = book.mid
    if mid is None:
        return np.nan
    side = Side.YES if side_yes else Side.NO
    vwap = book.vwap(side, Action.BUY, qty)
    if vwap is None:
        return np.nan
    ref = mid if side_yes else PAYOUT - mid
    return float(vwap - ref)


def liquidity_profile(book: OrderBook, sizes: Sequence[int] = (10, 50, 100, 500)) -> pd.DataFrame:
    """Slippage curve: how expensive it gets to demand size on each side."""
    rows = []
    for q in sizes:
        rows.append(
            {
                "qty": q,
                "yes_slippage_cents": quote_slippage(book, q, True),
                "no_slippage_cents": quote_slippage(book, q, False),
            }
        )
    return pd.DataFrame(rows)
