"""Closed-form arbitrage detectors.

Four structures cover essentially all model-free edge available on a binary
event exchange. Each detector is a fast O(depth) scan intended to run on every
book update; the general LP in :mod:`kalshi_alpha.arbitrage.lp` catches the
residual combinations these miss, at higher cost.

===========================  ============================================
Structure                    Violated no-arbitrage condition
===========================  ============================================
``crossed_book``             ``bid_YES < ask_YES`` within one market
``dutch_book_under``         ``sum_i ask_i >= $1`` over an exhaustive group
``dutch_book_over``          ``sum_i bid_i <= $1`` over an exclusive group
``ladder_monotonicity``      ``P(X >= k)`` non-increasing in ``k``
``cross_venue``              law of one price across exchanges
===========================  ============================================

**Canonical leg form.** Every leg emitted here is a *buy*: shorting YES is
expressed as buying NO at the mirrored price, which is exactly how the matching
engine executes it and removes all ambiguity about collateral and settlement
cash flows.

**A note on the naive complement trade.** "Buy YES + buy NO for less than $1"
cannot happen inside a single Kalshi market: the YES ask is *defined* as
``100 - NO bid``, so the pair costs ``100 + spread``, never less. Any hit from
``detect_crossed_book`` therefore signals a crossed or stale book -- valuable
as a data-integrity alarm, and occasionally real for a few milliseconds after
a large sweep. The genuine complement edge lives across venues, which is what
``detect_cross_venue`` is for.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from kalshi_alpha.arbitrage.fees import DEFAULT_TAKER_RATE, exact_taker_fee_cents
from kalshi_alpha.arbitrage.liquidity import Sweep, max_executable, sweep
from kalshi_alpha.arbitrage.payoff import (
    build_opportunity,
    exclusive_settle_map,
    ladder_settle_map,
)
from kalshi_alpha.config import ArbConfig
from kalshi_alpha.types import (
    PAYOUT,
    Action,
    ArbOpportunity,
    EventGroup,
    LadderGroup,
    Leg,
    OrderBook,
    Side,
)

BookMap = Mapping[str, OrderBook]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def legs_from_sweep(
    ticker: str, side: Side, action: Action, sw: Sweep, taker_rate: float = DEFAULT_TAKER_RATE
) -> list[Leg]:
    """Expand a book walk into one leg per price level actually consumed."""
    return [
        Leg(
            ticker=ticker,
            side=side,
            action=action,
            qty=qty,
            price=price,
            fee_cents=exact_taker_fee_cents(price, qty, taker_rate),
        )
        for price, qty in sw.levels
    ]


def _best_size(
    build: Callable[[int], list[Leg] | None],
    settle_map: Mapping[str, Sequence[int]],
    max_qty: int,
    min_edge: int,
    grid_points: int = 24,
) -> tuple[list[Leg], int] | None:
    """Search the quantity that maximises guaranteed profit.

    Payoff is linear in size and gross cost is convex (levels are consumed
    best-first), so guaranteed profit is concave -- *except* for a sawtooth of
    up to one cent per leg introduced by the per-order fee ceiling. A naive
    "stop at the first decline" scan gets trapped in that sawtooth and
    systematically under-sizes; a full 1..max_qty scan is exact but costs a
    book walk per quantity.

    We instead sweep a coarse grid, take the best point, and refine with a
    dense scan of one grid cell either side. That is O(grid + 2*cell)
    evaluations regardless of how deep the book is, and the concavity of the
    underlying curve guarantees the refined optimum is within the sawtooth
    amplitude of the true one.
    """
    from kalshi_alpha.arbitrage.payoff import pnl_over_outcomes

    if max_qty < 1:
        return None

    cache: dict[int, tuple[list[Leg], int] | None] = {}

    def evaluate(q: int) -> tuple[list[Leg], int] | None:
        if q in cache:
            return cache[q]
        legs = build(q)
        got = None
        if legs is not None:
            got = (legs, int(pnl_over_outcomes(legs, settle_map).min()))
        cache[q] = got
        return got

    step = max(1, max_qty // grid_points)
    grid = sorted({1, *range(step, max_qty + 1, step), max_qty})

    best: tuple[list[Leg], int] | None = None
    best_q = 1
    for q in grid:
        got = evaluate(q)
        if got is None:
            break  # book exhausted; deeper quantities cannot fill either
        if best is None or got[1] > best[1]:
            best, best_q = got, q

    if best is None:
        return None

    lo = max(1, best_q - step)
    hi = min(max_qty, best_q + step)
    for q in range(lo, hi + 1):
        got = evaluate(q)
        if got is None:
            break
        if got[1] > best[1]:
            best = got

    if best[1] < min_edge:
        return None
    return best


# --------------------------------------------------------------------------
# 1. within-market crossed book
# --------------------------------------------------------------------------
def detect_crossed_book(
    book: OrderBook, cfg: ArbConfig | None = None, taker_rate: float = DEFAULT_TAKER_RATE
) -> list[ArbOpportunity]:
    """Buy YES and buy NO for a combined cost below $1.

    Equivalent to the YES book being crossed. Fires on stale snapshots and, in
    live trading, on the sub-second window after a sweep clears one side.
    """
    cfg = cfg or ArbConfig()
    bid, ask = book.best_yes_bid, book.best_yes_ask
    if bid is None or ask is None or bid < ask:
        return []

    settle_map = {book.ticker: [1, 0]}
    cap = min(
        cfg.max_qty,
        max_executable(book, Side.YES, Action.BUY, cfg.size_haircut, cfg.depth_levels),
        max_executable(book, Side.NO, Action.BUY, cfg.size_haircut, cfg.depth_levels),
    )
    if cap < cfg.min_qty:
        return []

    def build(q: int) -> list[Leg] | None:
        yes = sweep(book, Side.YES, Action.BUY, q, taker_rate, cfg.size_haircut, cfg.depth_levels)
        no = sweep(book, Side.NO, Action.BUY, q, taker_rate, cfg.size_haircut, cfg.depth_levels)
        if not (yes.complete and no.complete):
            return None
        return legs_from_sweep(book.ticker, Side.YES, Action.BUY, yes, taker_rate) + legs_from_sweep(
            book.ticker, Side.NO, Action.BUY, no, taker_rate
        )

    found = _best_size(build, settle_map, cap, cfg.min_edge_cents)
    if found is None:
        return []
    legs, _ = found
    opp = build_opportunity(
        "crossed_book",
        book.ticker,
        legs,
        settle_map,
        ts=book.ts,
        detail=f"YES bid {bid}c >= YES ask {ask}c",
    )
    return [opp] if opp else []


# --------------------------------------------------------------------------
# 2/3. Dutch book across a mutually exclusive group
# --------------------------------------------------------------------------
def detect_dutch_book(
    group: EventGroup,
    books: BookMap,
    cfg: ArbConfig | None = None,
    taker_rate: float = DEFAULT_TAKER_RATE,
) -> list[ArbOpportunity]:
    """Under-round and over-round arbitrage across a mutually exclusive group.

    * **Under-round** (``sum ask < $1``): buy one YES of every outcome. Exactly
      one pays $1. Requires the group to be *exhaustive* -- if some outcome is
      missing, the portfolio can pay nothing and this is a punt, not an arb.
    * **Over-round** (``sum bid > $1``): buy one NO of every outcome. At least
      ``n-1`` pay $1. Needs only mutual exclusivity, so it is available on
      partial groups too.
    """
    cfg = cfg or ArbConfig()
    tickers = [t for t in group.tickers if t in books]
    if len(tickers) < 2 or len(tickers) > cfg.max_legs:
        return []
    if cfg.require_two_sided and not all(books[t].is_two_sided for t in tickers):
        return []

    settle_map = exclusive_settle_map(tickers, exhaustive=group.exhaustive)
    out: list[ArbOpportunity] = []

    def make_builder(side: Side):
        def build(q: int) -> list[Leg] | None:
            legs: list[Leg] = []
            for tk in tickers:
                sw = sweep(
                    books[tk], side, Action.BUY, q, taker_rate, cfg.size_haircut, cfg.depth_levels
                )
                if not sw.complete:
                    return None
                legs.extend(legs_from_sweep(tk, side, Action.BUY, sw, taker_rate))
            return legs

        return build

    # ---- under-round: buy every YES
    if group.exhaustive:
        cap = min(
            [cfg.max_qty]
            + [
                max_executable(books[t], Side.YES, Action.BUY, cfg.size_haircut, cfg.depth_levels)
                for t in tickers
            ]
        )
        if cap >= cfg.min_qty:
            found = _best_size(make_builder(Side.YES), settle_map, cap, cfg.min_edge_cents)
            if found:
                legs, _ = found
                total_ask = sum(books[t].best_yes_ask or PAYOUT for t in tickers)
                opp = build_opportunity(
                    "dutch_book_under",
                    group.event_ticker,
                    legs,
                    settle_map,
                    ts=max(books[t].ts for t in tickers),
                    detail=f"sum of best asks = {total_ask}c over {len(tickers)} outcomes",
                )
                if opp:
                    out.append(opp)

    # ---- over-round: buy every NO
    cap = min(
        [cfg.max_qty]
        + [
            max_executable(books[t], Side.NO, Action.BUY, cfg.size_haircut, cfg.depth_levels)
            for t in tickers
        ]
    )
    if cap >= cfg.min_qty:
        found = _best_size(make_builder(Side.NO), settle_map, cap, cfg.min_edge_cents)
        if found:
            legs, _ = found
            total_bid = sum(books[t].best_yes_bid or 0 for t in tickers)
            opp = build_opportunity(
                "dutch_book_over",
                group.event_ticker,
                legs,
                settle_map,
                ts=max(books[t].ts for t in tickers),
                detail=f"sum of best bids = {total_bid}c over {len(tickers)} outcomes",
            )
            if opp:
                out.append(opp)

    return out


# --------------------------------------------------------------------------
# 4. strike-ladder monotonicity
# --------------------------------------------------------------------------
def detect_ladder_violation(
    ladder: LadderGroup,
    books: BookMap,
    cfg: ArbConfig | None = None,
    taker_rate: float = DEFAULT_TAKER_RATE,
) -> list[ArbOpportunity]:
    """Vertical-spread arbitrage from a broken monotone ladder.

    For a ``>= k`` ladder the survival function must be non-increasing, so for
    ``k_i < k_j`` the pair ``BUY YES(k_i) + BUY NO(k_j)`` pays at least $1 in
    every state and $2 when the underlying lands between the strikes. It is an
    arbitrage whenever the pair costs less than $1 all-in.

    The scan is over **all** pairs, not just adjacent ones: with bid/ask
    spreads the executable condition ``ask_i < bid_j`` is not transitive, so a
    non-adjacent pair can be violated while every adjacent pair looks clean.

    Note that digital ladders require only *first-order* monotonicity. Unlike
    vanilla option strikes there is no butterfly/convexity constraint to
    enforce, because the second difference of a survival function is an
    unconstrained density increment.
    """
    cfg = cfg or ArbConfig()
    idx = [i for i, t in enumerate(ladder.tickers) if t in books]
    if len(idx) < 2:
        return []

    settle_map = ladder_settle_map(ladder.tickers, ladder.strikes, ladder.direction)
    out: list[ArbOpportunity] = []

    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            lo, hi = ladder.tickers[idx[a]], ladder.tickers[idx[b]]
            # For a decreasing ladder we need P(lo) >= P(hi): go long the low
            # strike and short the high strike. For an increasing ladder the
            # roles swap.
            long_tk, short_tk = (lo, hi) if ladder.decreasing else (hi, lo)
            bl, bs = books[long_tk], books[short_tk]
            ask = bl.best_yes_ask
            bid = bs.best_yes_bid
            if ask is None or bid is None or ask >= bid:
                continue  # no ordering violation at the top of book

            cap = min(
                cfg.max_qty,
                max_executable(bl, Side.YES, Action.BUY, cfg.size_haircut, cfg.depth_levels),
                max_executable(bs, Side.NO, Action.BUY, cfg.size_haircut, cfg.depth_levels),
            )
            if cap < cfg.min_qty:
                continue

            def build(q: int, bl: OrderBook = bl, bs: OrderBook = bs,
                      long_tk: str = long_tk, short_tk: str = short_tk) -> list[Leg] | None:
                sy = sweep(bl, Side.YES, Action.BUY, q, taker_rate, cfg.size_haircut,
                           cfg.depth_levels)
                sn = sweep(bs, Side.NO, Action.BUY, q, taker_rate, cfg.size_haircut,
                           cfg.depth_levels)
                if not (sy.complete and sn.complete):
                    return None
                return legs_from_sweep(long_tk, Side.YES, Action.BUY, sy, taker_rate) + \
                    legs_from_sweep(short_tk, Side.NO, Action.BUY, sn, taker_rate)

            found = _best_size(build, settle_map, cap, cfg.min_edge_cents)
            if not found:
                continue
            legs, _ = found
            opp = build_opportunity(
                "ladder_monotonicity",
                ladder.event_ticker,
                legs,
                settle_map,
                ts=max(bl.ts, bs.ts),
                detail=(
                    f"{long_tk} ask {ask}c < {short_tk} bid {bid}c "
                    f"(strikes {ladder.strikes[idx[a]]} vs {ladder.strikes[idx[b]]})"
                ),
            )
            if opp:
                out.append(opp)

    return out


# --------------------------------------------------------------------------
# 5. cross-venue law of one price
# --------------------------------------------------------------------------
def detect_cross_venue(
    book: OrderBook,
    other_yes_bid: int | None,
    other_yes_ask: int | None,
    venue: str = "other",
    cfg: ArbConfig | None = None,
    taker_rate: float = DEFAULT_TAKER_RATE,
    other_fee_per_contract: float = 0.0,
    other_depth: int = 0,
) -> list[ArbOpportunity]:
    """Buy the same claim cheap on one venue and short it dear on the other.

    The second venue is modelled as a top-of-book quote plus a flat per-contract
    fee, which is enough to cover Polymarket (gas + spread), an offshore book,
    or the implied probability from a listed option. Because the two venues
    settle on the same underlying event, the combined position pays exactly $1
    in every state.
    """
    cfg = cfg or ArbConfig()
    if other_depth < cfg.min_qty:
        return []
    settle_map = {book.ticker: [1, 0]}
    out: list[ArbOpportunity] = []

    # Leg A: buy YES on Kalshi, short YES (= buy NO) on the other venue.
    if other_yes_bid is not None:
        cap = min(
            cfg.max_qty,
            other_depth,
            max_executable(book, Side.YES, Action.BUY, cfg.size_haircut, cfg.depth_levels),
        )

        def build_a(q: int) -> list[Leg] | None:
            sw = sweep(book, Side.YES, Action.BUY, q, taker_rate, cfg.size_haircut,
                       cfg.depth_levels)
            if not sw.complete:
                return None
            legs = legs_from_sweep(book.ticker, Side.YES, Action.BUY, sw, taker_rate)
            legs.append(
                Leg(
                    ticker=book.ticker,
                    side=Side.NO,
                    action=Action.BUY,
                    qty=q,
                    price=PAYOUT - int(other_yes_bid),
                    fee_cents=int(round(other_fee_per_contract * q)),
                )
            )
            return legs

        if cap >= cfg.min_qty:
            found = _best_size(build_a, settle_map, cap, cfg.min_edge_cents)
            if found:
                opp = build_opportunity(
                    "cross_venue",
                    book.ticker,
                    found[0],
                    settle_map,
                    ts=book.ts,
                    detail=f"kalshi ask {book.best_yes_ask}c vs {venue} bid {other_yes_bid}c",
                )
                if opp:
                    out.append(opp)

    # Leg B: buy YES on the other venue, buy NO on Kalshi.
    if other_yes_ask is not None:
        cap = min(
            cfg.max_qty,
            other_depth,
            max_executable(book, Side.NO, Action.BUY, cfg.size_haircut, cfg.depth_levels),
        )

        def build_b(q: int) -> list[Leg] | None:
            sw = sweep(book, Side.NO, Action.BUY, q, taker_rate, cfg.size_haircut, cfg.depth_levels)
            if not sw.complete:
                return None
            legs = legs_from_sweep(book.ticker, Side.NO, Action.BUY, sw, taker_rate)
            legs.append(
                Leg(
                    ticker=book.ticker,
                    side=Side.YES,
                    action=Action.BUY,
                    qty=q,
                    price=int(other_yes_ask),
                    fee_cents=int(round(other_fee_per_contract * q)),
                )
            )
            return legs

        if cap >= cfg.min_qty:
            found = _best_size(build_b, settle_map, cap, cfg.min_edge_cents)
            if found:
                opp = build_opportunity(
                    "cross_venue",
                    book.ticker,
                    found[0],
                    settle_map,
                    ts=book.ts,
                    detail=f"{venue} ask {other_yes_ask}c vs kalshi bid {book.best_yes_bid}c",
                )
                if opp:
                    out.append(opp)

    return out


def scan_all(
    books: BookMap,
    groups: Iterable[EventGroup] = (),
    ladders: Iterable[LadderGroup] = (),
    cfg: ArbConfig | None = None,
    taker_rate: float = DEFAULT_TAKER_RATE,
) -> list[ArbOpportunity]:
    """Run every closed-form detector and return opportunities, richest first."""
    cfg = cfg or ArbConfig()
    found: list[ArbOpportunity] = []
    for book in books.values():
        found.extend(detect_crossed_book(book, cfg, taker_rate))
    for group in groups:
        found.extend(detect_dutch_book(group, books, cfg, taker_rate))
    for ladder in ladders:
        found.extend(detect_ladder_violation(ladder, books, cfg, taker_rate))
    found.sort(key=lambda o: o.worst_case_pnl_cents, reverse=True)
    return found
