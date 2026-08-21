"""Scheduled-release calendar.

The diffusion study needs release timestamps accurate to the second, and it
needs them from a source that is fixed *before* the price data is looked at.
Selecting event times after seeing the prices is the classic way to manufacture
a result out of nothing, so the calendar is a first-class, serialisable object:
it is written down, it is versioned with the repo, and the study consumes it
without ever consulting the price series.

Release times are stored in UTC. US macro releases are published at 08:30 or
10:00 US/Eastern, which moves in UTC across the daylight-saving boundary; the
helper here takes explicit UTC offsets so nothing silently shifts by an hour
twice a year -- a bug that would corrupt exactly the sub-minute window the
half-life estimates live in.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ScheduledRelease:
    """One scheduled information event."""

    ts: float  # epoch seconds, UTC
    name: str
    category: str = "macro"
    consensus: float | None = None
    actual: float | None = None
    importance: int = 3  # 1 (minor) .. 5 (market-moving)

    @property
    def surprise(self) -> float | None:
        """Actual minus consensus; the exogenous shock size."""
        if self.actual is None or self.consensus is None:
            return None
        return float(self.actual - self.consensus)

    @property
    def datetime_utc(self) -> datetime:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc)

    def describe(self) -> str:
        s = self.surprise
        tail = "" if s is None else f"  surprise={s:+.3f}"
        return f"{self.datetime_utc:%Y-%m-%d %H:%M:%S} UTC  {self.name}{tail}"


@dataclass
class EventCalendar:
    """A collection of releases, queryable by time and category."""

    releases: list[ScheduledRelease] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.releases)

    def __iter__(self):
        return iter(self.releases)

    def add(self, release: ScheduledRelease) -> None:
        self.releases.append(release)
        self.releases.sort(key=lambda r: r.ts)

    def timestamps(self, category: str | None = None, min_importance: int = 1) -> list[float]:
        return [
            r.ts
            for r in self.releases
            if (category is None or r.category == category) and r.importance >= min_importance
        ]

    def between(self, start_ts: float, end_ts: float) -> EventCalendar:
        return EventCalendar([r for r in self.releases if start_ts <= r.ts <= end_ts])

    def within(self, ts: float, window_s: float) -> list[ScheduledRelease]:
        return [r for r in self.releases if abs(r.ts - ts) <= window_s]

    def is_quiet(self, ts: float, window_s: float = 1_800.0) -> bool:
        """True when no release lands within ``window_s`` -- the safe window to quote in."""
        return not self.within(ts, window_s)

    # ---- persistence ---------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([asdict(r) for r in self.releases], indent=2), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> EventCalendar:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([ScheduledRelease(**r) for r in raw])

    @classmethod
    def from_iterable(cls, items: Iterable[ScheduledRelease]) -> EventCalendar:
        return cls(sorted(items, key=lambda r: r.ts))


def utc_ts(year: int, month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> float:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp()


def monthly_series(
    start: datetime,
    count: int,
    name: str,
    category: str = "macro",
    importance: int = 4,
    day_of_month: int = 12,
) -> list[ScheduledRelease]:
    """Approximate a recurring monthly release, for demos and tests.

    Real studies must use the published calendar; this exists so the offline
    demo has something structurally realistic to work with and is deliberately
    labelled as synthetic in the report.
    """
    out: list[ScheduledRelease] = []
    cur = start
    for _ in range(count):
        stamp = cur.replace(day=min(day_of_month, 28))
        out.append(
            ScheduledRelease(ts=stamp.timestamp(), name=name, category=category,
                             importance=importance)
        )
        cur = (stamp.replace(day=1) + timedelta(days=32)).replace(day=1)
    return out


def default_calendar(start_ts: float, n_events: int = 6, spacing_s: float = 86_400.0,
                     name: str = "SIM-RELEASE") -> EventCalendar:
    """Evenly spaced synthetic releases used by the offline demo."""
    return EventCalendar(
        [
            ScheduledRelease(ts=start_ts + i * spacing_s, name=f"{name}-{i:02d}",
                             category="simulated", importance=4)
            for i in range(n_events)
        ]
    )


def calendar_from_indices(
    times: Sequence[float], indices: Sequence[int], name: str = "SIM-RELEASE"
) -> EventCalendar:
    """Build a calendar from step indices into a simulated time axis."""
    return EventCalendar(
        [
            ScheduledRelease(ts=float(times[i]), name=f"{name}-{k:02d}", category="simulated",
                             importance=4)
            for k, i in enumerate(indices)
            if 0 <= i < len(times)
        ]
    )
