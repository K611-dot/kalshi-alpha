"""End-to-end CLI behaviour, configuration, storage round-trips and reporting."""

from __future__ import annotations

import json

import pytest

from kalshi_alpha.cli import backtest_table, build_parser, main, run_backtests
from kalshi_alpha.config import Settings, load_settings
from kalshi_alpha.data.events import (
    EventCalendar,
    ScheduledRelease,
    calendar_from_indices,
    default_calendar,
    utc_ts,
)
from kalshi_alpha.data.store import (
    TickStore,
    books_to_frame,
    frame_to_books,
    frame_to_trades,
    trades_to_frame,
)
from kalshi_alpha.report.html import Report, frame_to_html


class TestConfig:
    def test_defaults_are_offline(self) -> None:
        s = Settings()
        assert s.mode == "offline"
        assert not s.has_credentials

    def test_rest_base_switches_with_environment(self) -> None:
        assert "demo" in Settings(env="demo").rest_base
        assert "demo" not in Settings(env="prod").rest_base

    def test_invalid_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            Settings(mode="wat")

    def test_negative_fee_rate_is_rejected(self) -> None:
        from kalshi_alpha.config import FeeConfig

        with pytest.raises(ValueError):
            FeeConfig(taker_rate=-0.1)

    def test_env_overrides_apply(self, monkeypatch) -> None:
        monkeypatch.setenv("KALSHI_ENV", "prod")
        monkeypatch.setenv("KALSHI_ALPHA_LOG_LEVEL", "DEBUG")
        s = load_settings()
        assert s.env == "prod"
        assert s.log_level == "DEBUG"

    def test_explicit_override_beats_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("KALSHI_ENV", "prod")
        assert load_settings(env="demo").env == "demo"

    def test_settings_serialise(self) -> None:
        payload = json.loads(Settings().model_dump_json())
        assert payload["fees"]["taker_rate"] == pytest.approx(0.07)


class TestCalendar:
    def test_surprise_requires_both_sides(self) -> None:
        assert ScheduledRelease(0.0, "x").surprise is None
        assert ScheduledRelease(0.0, "x", consensus=2.0, actual=2.5).surprise == 0.5

    def test_calendar_stays_sorted(self) -> None:
        cal = EventCalendar()
        cal.add(ScheduledRelease(200.0, "b"))
        cal.add(ScheduledRelease(100.0, "a"))
        assert [r.name for r in cal] == ["a", "b"]

    def test_quiet_window_detection(self) -> None:
        cal = default_calendar(1_000_000.0, n_events=3, spacing_s=10_000.0)
        assert not cal.is_quiet(1_000_000.0, window_s=100.0)
        assert cal.is_quiet(1_005_000.0, window_s=100.0)

    def test_filtering_by_importance(self) -> None:
        cal = EventCalendar([
            ScheduledRelease(1.0, "minor", importance=1),
            ScheduledRelease(2.0, "major", importance=5),
        ])
        assert cal.timestamps(min_importance=3) == [2.0]

    def test_json_round_trip(self, tmp_path) -> None:
        cal = default_calendar(utc_ts(2026, 3, 1, 13, 30), n_events=4)
        path = tmp_path / "cal.json"
        cal.to_json(path)
        assert len(EventCalendar.from_json(path)) == 4

    def test_calendar_from_indices(self) -> None:
        times = [float(i) for i in range(100)]
        cal = calendar_from_indices(times, [10, 50, 900])
        assert len(cal) == 2  # the out-of-range index is dropped


class TestStore:
    def test_book_round_trip_preserves_the_touch(self, sim) -> None:
        books = [snap[sim.tickers[0]] for snap in sim.books[:50]]
        restored = frame_to_books(books_to_frame(books))
        assert len(restored) == len(books)
        for a, b in zip(books, restored, strict=True):
            assert a.best_yes_bid == b.best_yes_bid
            assert a.best_yes_ask == b.best_yes_ask
            assert a.ticker == b.ticker

    def test_trade_round_trip(self, sim) -> None:
        trades = sim.trades[sim.tickers[0]][:40]
        if not trades:
            pytest.skip("no trades generated for this ticker")
        restored = frame_to_trades(trades_to_frame(trades))
        assert [t.signed_size for t in restored] == [t.signed_size for t in trades]

    def test_parquet_partitioning_and_read_back(self, tmp_path, sim) -> None:
        store = TickStore(tmp_path / "parquet")
        books = [snap[sim.tickers[0]] for snap in sim.books[:80]]
        assert store.write_books(books, "EVT") is not None
        assert store.write_trades(sim.trades[sim.tickers[0]][:20], "EVT") is not None
        back = store.read_books("EVT")
        assert len(back) == 80
        assert store.events("books") == ["EVT"]
        assert store.stats()["books_files"] == 1

    def test_reading_a_missing_event_is_empty_not_an_error(self, tmp_path) -> None:
        assert TickStore(tmp_path).read_books("NOPE").empty

    def test_writing_nothing_is_a_no_op(self, tmp_path) -> None:
        assert TickStore(tmp_path).write_books([], "EVT") is None


class TestReport:
    def test_report_is_self_contained(self) -> None:
        report = Report("t", "s")
        report.section("A").text("body").note("note").pre("code")
        html = report.render()
        assert "<style>" in html
        assert "http://" not in html and "https://" not in html

    def test_html_escaping(self) -> None:
        report = Report("t")
        report.section("A").text("<script>alert(1)</script>")
        assert "<script>" not in report.render()

    def test_empty_frame_renders_a_placeholder(self) -> None:
        import pandas as pd

        assert "No rows" in frame_to_html(pd.DataFrame())

    def test_write_creates_the_file(self, tmp_path) -> None:
        report = Report("t")
        report.section("A").text("x")
        path = report.write(tmp_path / "nested" / "r.html")
        assert path.exists() and path.read_text(encoding="utf-8").startswith("<!doctype")


class TestCLI:
    def test_parser_requires_a_command(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_scan_runs_offline(self, capsys) -> None:
        assert main(["scan", "--steps", "300"]) == 0
        assert "scanned" in capsys.readouterr().out

    def test_scan_with_a_dislocation_finds_something(self, capsys) -> None:
        assert main(["scan", "--steps", "300", "--dislocate", "12"]) == 0
        assert "ladder_monotonicity" in capsys.readouterr().out

    def test_config_prints_valid_json(self, capsys) -> None:
        assert main(["config"]) == 0
        assert json.loads(capsys.readouterr().out)["mode"] == "offline"

    def test_calibrate_runs(self, capsys) -> None:
        assert main(["calibrate", "--n", "2000"]) == 0
        assert "brier" in capsys.readouterr().out

    def test_backtest_all_strategies(self, capsys) -> None:
        assert main(["backtest", "--strategy", "all", "--steps", "400"]) == 0
        assert "ladder_arb" in capsys.readouterr().out

    @pytest.mark.slow
    def test_demo_writes_a_report(self, tmp_path) -> None:
        assert main(["demo", "--out", str(tmp_path), "--steps", "400"]) == 0
        report = tmp_path / "report.html"
        assert report.exists()
        assert report.stat().st_size > 10_000
        for name in ("backtests.csv", "halflife_recovery.csv",
                     "information_share_recovery.csv", "violation_threshold.csv"):
            assert (tmp_path / name).exists()


def test_backtest_table_covers_every_strategy(sim) -> None:
    table = backtest_table(run_backtests(sim, Settings(), n_trials=4))
    assert set(table["strategy"]) >= {"ladder_arb", "coherence", "microprice", "drift"}
    assert "deflated_sr" in table.columns
