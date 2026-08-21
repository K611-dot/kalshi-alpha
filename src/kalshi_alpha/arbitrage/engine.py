"""Scanner orchestration.

Runs the cheap closed-form detectors first, then the general LP on any group
they did not already resolve, deduplicates overlapping claims, and reports how
long each stage took. Latency numbers matter here: an opportunity that takes
40 ms to find is an opportunity somebody else has already taken.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from kalshi_alpha.arbitrage.detectors import (
    detect_crossed_book,
    detect_dutch_book,
    detect_ladder_violation,
)
from kalshi_alpha.arbitrage.fees import DEFAULT_TAKER_RATE
from kalshi_alpha.arbitrage.lp import find_arbitrage, implied_probabilities
from kalshi_alpha.arbitrage.payoff import (
    exclusive_settle_map,
    ladder_settle_map,
    verify_opportunity,
)
from kalshi_alpha.config import Settings
from kalshi_alpha.logging_setup import get_logger
from kalshi_alpha.types import ArbOpportunity, EventGroup, LadderGroup, OrderBook

log = get_logger(__name__)


@dataclass
class ScanResult:
    opportunities: list[ArbOpportunity] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    implied: dict[str, dict[str, float]] = field(default_factory=dict)
    n_books: int = 0
    n_groups: int = 0
    n_ladders: int = 0
    n_rejected: int = 0

    @property
    def total_guaranteed_cents(self) -> int:
        return sum(o.worst_case_pnl_cents for o in self.opportunities)

    @property
    def total_capital_cents(self) -> int:
        return sum(o.capital_at_risk_cents for o in self.opportunities)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "kind": o.kind,
                "event": o.event_ticker,
                "legs": len(o.legs),
                "qty": o.qty,
                "cost_cents": o.cost_cents,
                "guaranteed_cents": o.worst_case_pnl_cents,
                "best_case_cents": o.best_case_pnl_cents,
                "roc": o.return_on_capital,
                "detail": o.detail,
            }
            for o in self.opportunities
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            f"scanned {self.n_books} books / {self.n_groups} groups / {self.n_ladders} ladders",
            f"found {len(self.opportunities)} opportunities, "
            f"{self.total_guaranteed_cents}c guaranteed on "
            f"{self.total_capital_cents}c capital",
            "timings: " + ", ".join(f"{k}={v:.2f}ms" for k, v in self.timings_ms.items()),
        ]
        return "\n".join(lines)


def _leg_signature(opp: ArbOpportunity) -> tuple:
    return tuple(sorted((lg.ticker, lg.side.value, lg.action.value, lg.price) for lg in opp.legs))


def dedupe(opps: Sequence[ArbOpportunity]) -> list[ArbOpportunity]:
    """Keep the richest opportunity per distinct set of legs.

    The LP frequently rediscovers what a closed-form detector already found; we
    keep whichever version books more guaranteed profit, since acting on both
    would double-count the same liquidity.
    """
    best: dict[tuple, ArbOpportunity] = {}
    for opp in opps:
        sig = _leg_signature(opp)
        cur = best.get(sig)
        if cur is None or opp.worst_case_pnl_cents > cur.worst_case_pnl_cents:
            best[sig] = opp
    return sorted(best.values(), key=lambda o: o.worst_case_pnl_cents, reverse=True)


class ArbEngine:
    """Stateless scanner. Safe to call on every book update."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.cfg = self.settings.arb
        self.taker_rate = self.settings.fees.taker_rate

    # ---- individual stages -------------------------------------------
    def scan_books(self, books: Mapping[str, OrderBook]) -> list[ArbOpportunity]:
        out: list[ArbOpportunity] = []
        for book in books.values():
            out.extend(detect_crossed_book(book, self.cfg, self.taker_rate))
        return out

    def scan_groups(
        self, groups: Iterable[EventGroup], books: Mapping[str, OrderBook]
    ) -> list[ArbOpportunity]:
        out: list[ArbOpportunity] = []
        for group in groups:
            out.extend(detect_dutch_book(group, books, self.cfg, self.taker_rate))
        return out

    def scan_ladders(
        self, ladders: Iterable[LadderGroup], books: Mapping[str, OrderBook]
    ) -> list[ArbOpportunity]:
        out: list[ArbOpportunity] = []
        for ladder in ladders:
            out.extend(detect_ladder_violation(ladder, books, self.cfg, self.taker_rate))
        return out

    def scan_lp(
        self,
        books: Mapping[str, OrderBook],
        groups: Iterable[EventGroup],
        ladders: Iterable[LadderGroup],
    ) -> tuple[list[ArbOpportunity], dict[str, dict[str, float]]]:
        out: list[ArbOpportunity] = []
        implied: dict[str, dict[str, float]] = {}
        for group in groups:
            tickers = [t for t in group.tickers if t in books]
            if len(tickers) < 2:
                continue
            smap = exclusive_settle_map(tickers, group.exhaustive)
            sol = find_arbitrage(books, smap, group.event_ticker, self.cfg, self.taker_rate)
            if sol.opportunity:
                out.append(sol.opportunity)
            else:
                probs = implied_probabilities(books, smap, self.cfg, self.taker_rate)
                if probs:
                    implied[group.event_ticker] = probs
        for ladder in ladders:
            tickers = [t for t in ladder.tickers if t in books]
            if len(tickers) < 2:
                continue
            smap = ladder_settle_map(ladder.tickers, ladder.strikes, ladder.direction)
            smap = {k: v for k, v in smap.items() if k in books}
            sol = find_arbitrage(books, smap, ladder.event_ticker, self.cfg, self.taker_rate)
            if sol.opportunity:
                out.append(sol.opportunity)
            else:
                probs = implied_probabilities(books, smap, self.cfg, self.taker_rate)
                if probs:
                    implied[ladder.event_ticker] = probs
        return out, implied

    # ---- full scan ----------------------------------------------------
    def scan(
        self,
        books: Mapping[str, OrderBook],
        groups: Sequence[EventGroup] = (),
        ladders: Sequence[LadderGroup] = (),
        use_lp: bool = True,
    ) -> ScanResult:
        timings: dict[str, float] = {}
        found: list[ArbOpportunity] = []

        t0 = time.perf_counter()
        found += self.scan_books(books)
        timings["crossed_book"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        found += self.scan_groups(groups, books)
        timings["dutch_book"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        found += self.scan_ladders(ladders, books)
        timings["ladder"] = (time.perf_counter() - t0) * 1e3

        implied: dict[str, dict[str, float]] = {}
        if use_lp:
            t0 = time.perf_counter()
            lp_opps, implied = self.scan_lp(books, groups, ladders)
            found += lp_opps
            timings["lp"] = (time.perf_counter() - t0) * 1e3

        before = len(found)
        opps = dedupe(found)
        result = ScanResult(
            opportunities=opps,
            timings_ms=timings,
            implied=implied,
            n_books=len(books),
            n_groups=len(list(groups)),
            n_ladders=len(list(ladders)),
            n_rejected=before - len(opps),
        )
        if opps:
            log.info(
                "arb scan complete",
                extra={
                    "found": len(opps),
                    "guaranteed_cents": result.total_guaranteed_cents,
                    "scan_ms": round(sum(timings.values()), 2),
                },
            )
        return result

    # ---- validation ----------------------------------------------------
    def validate(
        self, opp: ArbOpportunity, settle_map: Mapping[str, Sequence[int]]
    ) -> bool:
        """Re-prove an opportunity immediately before sending orders."""
        ok = verify_opportunity(opp, settle_map)
        if not ok:
            log.warning("arb failed pre-trade validation", extra={"kind": opp.kind})
        return ok


def scan_offline(
    books: Mapping[str, OrderBook],
    groups: Sequence[EventGroup] = (),
    ladders: Sequence[LadderGroup] = (),
    settings: Settings | None = None,
) -> ScanResult:
    """Convenience wrapper used by the CLI and the tests."""
    return ArbEngine(settings).scan(books, groups, ladders)


__all__ = [
    "ArbEngine",
    "ScanResult",
    "dedupe",
    "scan_offline",
    "DEFAULT_TAKER_RATE",
]
