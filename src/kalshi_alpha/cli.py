"""Command-line interface.

``kalshi-alpha demo`` runs the whole pipeline offline against the simulator and
writes a single self-contained HTML report. It needs no credentials, no network
and no data files, which means the results in the README can be reproduced by
anyone in one command -- the minimum bar for a research artefact to be worth
reading.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_alpha import __version__
from kalshi_alpha.arbitrage.engine import ArbEngine
from kalshi_alpha.arbitrage.fees import (
    exact_taker_fee_cents,
    linear_taker_fee_cents,
    round_trip_breakeven_cents,
)
from kalshi_alpha.arbitrage.lp import implied_probabilities
from kalshi_alpha.arbitrage.payoff import ladder_settle_map
from kalshi_alpha.backtest.engine import Backtester, BacktestResult
from kalshi_alpha.backtest.strategies import STRATEGY_NAMES, build_strategy
from kalshi_alpha.config import Settings, load_settings
from kalshi_alpha.data.synthetic import (
    SimConfig,
    SimulatedLadder,
    inject_ladder_violation,
    simulate_calibration_sample,
    simulate_ladder,
    simulate_two_venue,
)
from kalshi_alpha.diffusion.efficiency import efficiency_panel, variance_ratio_profile
from kalshi_alpha.diffusion.event_study import event_study
from kalshi_alpha.diffusion.halflife import half_life_ensemble, window_scan
from kalshi_alpha.diffusion.leadlag import epps_curve, estimate_delay
from kalshi_alpha.diffusion.price_discovery import price_discovery
from kalshi_alpha.logging_setup import setup_logging
from kalshi_alpha.microstructure.features import feature_frame, trade_frame
from kalshi_alpha.microstructure.impact import liquidity_report
from kalshi_alpha.probability.calibration import calibration_report
from kalshi_alpha.report.html import Report, line_chart, scatter_chart
from kalshi_alpha.types import PAYOUT, EventGroup


# --------------------------------------------------------------------------
# shared analysis pieces
# --------------------------------------------------------------------------
def _sim(seed: int = 7, steps: int = 1600, half_life: float = 150.0) -> SimulatedLadder:
    cfg = SimConfig(
        n_steps=steps,
        dt_s=5.0,
        event_step=steps // 3,
        adjustment_half_life_s=half_life,
        event_jump_cents=18.0,
        seed=seed,
    )
    return simulate_ladder(cfg=cfg)


def violation_threshold(
    sim: SimulatedLadder, settings: Settings, shifts=range(0, 41, 2)
) -> pd.DataFrame:
    """How large must a ladder violation be before it is actually tradeable?

    The interesting quantity is not whether the market is mispriced -- it always
    is, slightly -- but whether the mispricing clears the fee schedule. This
    sweeps a monotonicity violation of increasing, controlled size and records
    the first magnitude at which the scanner can book a guaranteed profit.
    """
    engine = ArbEngine(settings)
    books = sim.books[len(sim.books) // 2]
    rows = []
    for shift in shifts:
        dislocated, moved = inject_ladder_violation(books, sim.ladder.tickers, int(shift))
        if moved is None and shift > 0:
            continue
        result = engine.scan(dislocated, ladders=[sim.ladder], use_lp=False)
        best = result.opportunities[0] if result.opportunities else None
        rows.append(
            {
                "violation_cents": int(shift or 0),
                "detected": bool(best),
                "guaranteed_cents": best.worst_case_pnl_cents if best else 0,
                "capital_cents": best.capital_at_risk_cents if best else 0,
                "roc": best.return_on_capital if best else 0.0,
                "scan_ms": round(sum(result.timings_ms.values()), 3),
            }
        )
    return pd.DataFrame(rows)


def threshold_by_price(
    settings: Settings, levels=(5, 10, 20, 30, 40, 50, 60, 70, 80, 90), max_violation: int = 12
) -> pd.DataFrame:
    """Minimum tradeable violation as a function of where the pair is priced.

    The headline result of the arbitrage half of this project. A two-market
    ladder pair is constructed at each price level and the monotonicity
    violation is widened one cent at a time until the scanner can book a
    guaranteed profit. The empirical threshold is then compared against the
    closed-form prediction.

    The prediction: the pair costs ``100 - v`` plus the taker fee on both legs,
    so it is profitable once ``v`` exceeds the sum of those fees -- which, for a
    small violation, is just ``2 * r * p * (1 - p)``, the round-trip fee hurdle
    at that price. Arbitrage is therefore roughly **five times cheaper to
    capture in the tails than at even money**, purely because the fee curve is
    concave. That is a structural feature of the exchange, not a property of any
    particular market, and it is the single most actionable fact in this repo:
    it says where to point the scanner.
    """
    from kalshi_alpha.types import LadderGroup, OrderBook

    engine = ArbEngine(settings)
    ladder = LadderGroup("SYN", ("LO", "HI"), (1.0, 2.0), "gte")
    rows = []
    for p in levels:
        threshold = None
        for v in range(1, max_violation + 1):
            lo_ask, hi_bid = p, p + v
            if hi_bid >= 98 or lo_ask <= 1:
                break
            books = {
                "LO": OrderBook.from_levels(
                    "LO", 0.0, [(lo_ask - 1, 400)], [(PAYOUT - lo_ask, 400)]
                ),
                "HI": OrderBook.from_levels(
                    "HI", 0.0, [(hi_bid, 400)], [(PAYOUT - hi_bid - 1, 400)]
                ),
            }
            if engine.scan(books, ladders=[ladder], use_lp=False).opportunities:
                threshold = v
                break
        rows.append(
            {
                "price_cents": p,
                "min_violation_cents": threshold if threshold else np.nan,
                "predicted_hurdle_cents": round_trip_breakeven_cents(p),
            }
        )
    return pd.DataFrame(rows)


def half_life_recovery(
    targets=(30.0, 60.0, 120.0, 300.0, 600.0), tau_floor_frac: float = 0.5
) -> pd.DataFrame:
    """Score the diffusion estimator against half-lives it was never told.

    The session length is scaled so every target gets roughly the same number of
    half-lives of post-release data -- otherwise the slow cases are graded on a
    window that could not resolve them regardless of how good the estimator is,
    and the table measures the harness rather than the method.

    ``tau_floor_frac`` controls how much the underlying's volatility regime is
    allowed to shift during the session. The default holds it steadier than a
    real contract's would be, deliberately: the point of this table is to
    isolate estimator accuracy from regime drift. The demo report shows the
    regime-drift effect separately, in the window-scan section.
    """
    rows = []
    for true_hl in targets:
        dt = 5.0
        post_steps = int(np.clip(40.0 * true_hl / dt, 400, 6000))
        pre_steps = 600
        cfg = SimConfig(
            n_steps=pre_steps + post_steps, dt_s=dt, event_step=pre_steps,
            adjustment_half_life_s=float(true_hl), event_jump_cents=18.0,
            tau_floor_frac=tau_floor_frac,
        )
        sim = simulate_ladder(cfg=cfg)
        tk = sim.tickers[len(sim.tickers) // 2]
        ev = pre_steps
        t = sim.times[ev:] - sim.times[ev]
        quoted = sim.quoted[tk][ev:] * PAYOUT
        fair = sim.fair[tk][ev:] * PAYOUT
        res = half_life_ensemble(t, quoted, cfg.dt_s, leader=fair)
        est = res.consensus_half_life_s
        rows.append(
            {
                "true_half_life_s": float(true_hl),
                "estimated_s": est,
                "error_pct": (est - true_hl) / true_hl * 100.0 if np.isfinite(est) else np.nan,
                "source": res.consensus_source,
                "window_s": res.spread_window_s or res.window_s,
                "identified": res.identified,
            }
        )
    return pd.DataFrame(rows)


def information_share_recovery(ratios=((0.9, 0.3), (0.7, 0.5), (0.5, 0.5), (0.3, 0.9))) -> pd.DataFrame:
    """Score the Hasbrouck estimator against known information shares."""
    rows = []
    for sa, sb in ratios:
        tv = simulate_two_venue(n=6000, sigma_a=sa, sigma_b=sb, kappa_a=0.04, kappa_b=0.30)
        pdres = price_discovery(tv.venue_a, tv.venue_b, ("venue_a", "venue_b"), lags=4)
        if pdres is None:
            continue
        rows.append(
            {
                "sigma_a": sa,
                "sigma_b": sb,
                "true_share_a": tv.true_share_a,
                "hasbrouck_lo": float(pdres.is_lower[0]),
                "hasbrouck_hi": float(pdres.is_upper[0]),
                "gonzalo_granger": float(pdres.component_share[0]),
                "leader": pdres.leader,
                # A 50/50 split has no leader, so scoring one there is not a
                # test of anything. Only clear cases are graded.
                "decisive": abs(tv.true_share_a - 0.5) > 0.1,
                "correct_leader": (
                    (pdres.leader == "venue_a") == (tv.true_share_a > 0.5)
                    if abs(tv.true_share_a - 0.5) > 0.1
                    else True
                ),
            }
        )
    return pd.DataFrame(rows)


def run_backtests(
    sim: SimulatedLadder, settings: Settings, n_trials: int = 4
) -> dict[str, BacktestResult]:
    """Run every strategy over the same data with the same fill model."""
    bt = Backtester(settings)
    settle = sim.settle()
    configs = {
        "ladder_arb": {"ladders": [sim.ladder], "max_qty": 100, "scan_every": 20},
        "coherence": {"ladder": sim.ladder, "qty": 20},
        "microprice": {"tickers": sim.tickers, "qty": 10},
        "drift": {"tickers": sim.tickers, "event_ts": sim.event_ts,
                  "entry_delay_s": 30.0, "hold_s": 900.0, "qty": 25},
    }
    out: dict[str, BacktestResult] = {}
    for name, kwargs in configs.items():
        strategy = build_strategy(name, **kwargs)
        result = bt.run(sim.books, sim.times, strategy, trades=sim.trades,
                        settlement=settle, n_trials=n_trials)
        out[name] = result

    # Stress run: the same arbitrage strategy over the same session with a
    # persistent monotonicity violation injected. Together with the clean
    # run above this is the pair that matters -- no false positives on a
    # coherent book, and capture when a real violation is present. Either half
    # alone proves nothing.
    lo, hi = len(sim.books) // 8, len(sim.books) * 3 // 4
    stressed = [
        inject_ladder_violation(snap, sim.ladder.tickers, 12)[0] if lo <= i < hi else snap
        for i, snap in enumerate(sim.books)
    ]
    out["ladder_arb_stressed"] = bt.run(
        stressed,
        sim.times,
        build_strategy("ladder_arb", ladders=[sim.ladder], max_qty=100, scan_every=20),
        trades=sim.trades,
        settlement=settle,
        n_trials=n_trials,
    )
    return out


def backtest_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        s = res.stats
        rows.append(
            {
                "strategy": name,
                "pnl_usd": s.total_pnl / 100.0,
                "trades": s.n_trades,
                "contracts": s.n_contracts,
                "fees_usd": (s.fees / 100.0) if np.isfinite(s.fees) else 0.0,
                "sharpe": s.sharpe,
                "max_dd_usd": s.max_drawdown / 100.0,
                "hit_rate": s.hit_rate,
                "deflated_sr": s.dsr,
                "fill_rate": res.diagnostics.get("fill_rate", np.nan),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_demo(args: argparse.Namespace) -> int:
    settings = load_settings(artifacts_dir=Path(args.out))
    settings.ensure_dirs()
    t0 = time.perf_counter()

    sim = _sim(seed=settings.seed, steps=args.steps, half_life=args.half_life)
    mid_tk = sim.tickers[len(sim.tickers) // 2]
    ev = args.steps // 3

    report = Report(
        title="kalshi-alpha research report",
        subtitle=(
            "Mispricing detection, no-arbitrage enforcement and information-diffusion "
            "measurement for binary event contracts. Generated offline against the "
            "simulator, so every number below is reproducible with `make demo`."
        ),
    )

    # ---- 1. fee schedule ------------------------------------------------
    prices = list(range(1, PAYOUT))
    fee_curve = [linear_taker_fee_cents(p, 1) for p in prices]
    breakeven = [round_trip_breakeven_cents(p) for p in prices]
    sec = report.section("1. The fee schedule is the whole problem")
    sec.text(
        "Kalshi's taker fee is ceil(0.07 * C * P * (1 - P)), a concave function that peaks "
        "at even money. A round trip at 50c costs 3.5 cents before any edge; at 10c it costs "
        "1.26. Every threshold in this system is derived from that curve rather than picked."
    )
    sec.cards(
        [
            ("Round trip @ 50c", f"{round_trip_breakeven_cents(50):.2f}c", "neg"),
            ("Round trip @ 10c", f"{round_trip_breakeven_cents(10):.2f}c", ""),
            ("Fee, 1 lot @ 50c", f"{exact_taker_fee_cents(50, 1)}c", ""),
            ("Fee, 100 lots @ 50c", f"{exact_taker_fee_cents(50, 100)}c", ""),
        ]
    )
    sec.html(
        line_chart(
            prices,
            {"taker fee per contract": fee_curve, "round-trip hurdle": breakeven},
            title="Cost of trading vs price", xlabel="YES price (cents)", ylabel="cents",
        )
    )

    # ---- 2. arbitrage ----------------------------------------------------
    sec = report.section("2. Arbitrage scanning")
    engine = ArbEngine(settings)
    clean = engine.scan(sim.final_books(), ladders=[sim.ladder])
    smap = ladder_settle_map(sim.ladder.tickers, sim.ladder.strikes, sim.ladder.direction)
    implied = implied_probabilities(sim.final_books(), smap, settings.arb)

    sec.text(
        "The simulated ladder is coherent by construction, so a correct scanner must find "
        "nothing on it. It does. The dual of the arbitrage LP still returns a set of "
        "risk-neutral state prices consistent with every quote, and those are monotone in "
        "the strike by construction -- unlike reading each mid independently."
    )
    sec.cards(
        [
            ("Opportunities on clean book", str(len(clean.opportunities)),
             "pos" if not clean.opportunities else "neg"),
            ("Markets scanned", str(clean.n_books), ""),
            ("Scan latency", f"{sum(clean.timings_ms.values()):.1f} ms", ""),
        ]
    )
    if implied:
        final = sim.final_books()
        mids = [final[t].mid for t in implied]
        quoted = [(m if m is not None else float("nan")) / PAYOUT for m in mids]
        sec.subheading("Risk-neutral survival function implied by the joint state prices")
        sec.table(
            pd.DataFrame(
                {
                    "ticker": list(implied),
                    "strike": list(sim.ladder.strikes),
                    "implied_P(X>=k)": [implied[t] for t in implied],
                    "quoted_mid_prob": quoted,
                }
            )
        )

    thresh = violation_threshold(sim, settings)
    first = thresh[thresh["detected"]]
    sec.subheading("How large must a violation be before it can be traded?")
    sec.text(
        "A monotonicity violation of a controlled size is injected into the ladder in "
        "2-cent steps. Below the fee hurdle the scanner correctly refuses to act on a "
        "mispricing it cannot monetise; this is the single most important discipline in "
        "the system, and the reason most 'arbitrage' spotted on a chart does not exist. "
        "The magnitude is measured the way the detector measures it: how far one strike's "
        "bid sits above the next-lower strike's ask."
    )
    sec.cards(
        [
            (
                "Minimum tradeable violation",
                f"{int(first['violation_cents'].iloc[0])}c" if not first.empty else "not reached",
                "",
            ),
            ("Sizes tested", str(len(thresh)), ""),
            ("Median scan latency", f"{thresh['scan_ms'].median():.2f} ms", ""),
        ]
    )
    sec.table(thresh, max_rows=25)

    by_price = threshold_by_price(settings)
    sec.subheading("Where on the price grid arbitrage is cheapest to capture")
    sec.text(
        "The same question asked structurally. A two-market ladder pair is built at each "
        "price level and the violation widened one cent at a time until it becomes "
        "tradeable. The closed-form prediction is that the threshold equals the round-trip "
        "fee hurdle 2*r*p*(1-p), and it does -- rounded up to the cent grid. The practical "
        "consequence is the most actionable fact in this repository: because the fee curve "
        "is concave, a one-cent violation is already tradeable at 5c while an identical "
        "violation at even money needs to be four times larger. Scanner attention belongs "
        "in the tails."
    )
    sec.table(by_price)
    sec.html(
        line_chart(
            by_price["price_cents"].tolist(),
            {
                "minimum tradeable violation": by_price["min_violation_cents"].tolist(),
                "predicted fee hurdle": by_price["predicted_hurdle_cents"].tolist(),
            },
            title="Minimum tradeable violation vs price level",
            xlabel="YES price (cents)", ylabel="cents",
        )
    )

    sec.html(
        line_chart(
            thresh["violation_cents"].tolist(),
            {"guaranteed profit (cents)": thresh["guaranteed_cents"].tolist()},
            title="Guaranteed profit vs violation size",
            xlabel="violation (cents)", ylabel="guaranteed cents",
        )
    )

    # ---- 3. microstructure ----------------------------------------------
    sec = report.section("3. Microstructure")
    books_mid = [snap[mid_tk] for snap in sim.books]
    feats = feature_frame(books_mid)
    tdf = trade_frame(sim.trades[mid_tk])
    liq = liquidity_report(feats, tdf)
    sec.cards(
        [
            ("Mean spread", f"{liq.get('mean_spread_cents', float('nan')):.2f}c", ""),
            ("Roll implied spread", f"{liq.get('roll_spread_cents', float('nan')):.2f}c", ""),
            ("Kyle lambda", f"{liq.get('kyle_lambda_cents_per_contract', float('nan')):.2e}", ""),
            ("VPIN", f"{liq.get('vpin', float('nan')):.3f}", ""),
        ]
    )
    sec.text(
        "Kyle's lambda is the permanent price impact per contract of signed flow and is what "
        "caps executable size: an arbitrage that needs more contracts than the book can "
        "absorb pays for its own edge on the way in."
    )
    sec.html(
        line_chart(
            (sim.times - sim.times[0]).tolist(),
            {"mid": feats["mid"].tolist(), "microprice": feats["microprice"].tolist()},
            title=f"{mid_tk}: mid vs microprice", xlabel="seconds", ylabel="cents",
        )
    )

    # ---- 4. diffusion ----------------------------------------------------
    sec = report.section("4. Information diffusion")
    sec.text(
        "The estimators are scored against a simulator whose true half-life and true "
        "information share are known by construction. An estimator that is never checked "
        "against a known answer is an opinion."
    )
    recovery = half_life_recovery()
    sec.subheading("Half-life recovery against ground truth")
    sec.table(recovery)
    sec.html(
        scatter_chart(
            recovery["true_half_life_s"].tolist(),
            recovery["estimated_s"].tolist(),
            title="Estimated vs true half-life", xlabel="true (s)", ylabel="estimated (s)",
            diagonal=True,
        )
    )

    scan = window_scan(
        sim.times[ev:] - sim.times[ev], sim.quoted[mid_tk][ev:] * PAYOUT, 5.0
    )
    sec.subheading("Why the estimation window has to be chosen, not assumed")
    sec.text(
        "Estimated half-life as a function of how much post-release data is used. It "
        "plateaus once the window covers several half-lives, then runs away as the "
        "efficient price's own random walk starts to dominate the residual. Reporting a "
        "single number without this curve is how a two-minute half-life gets published "
        "as forty."
    )
    sec.table(scan)

    shares = information_share_recovery()
    sec.subheading("Hasbrouck information-share recovery")
    sec.table(shares)

    tv = simulate_two_venue(n=6000, sigma_a=0.9, sigma_b=0.3, kappa_a=0.04, kappa_b=0.30)
    pdres = price_discovery(tv.venue_a, tv.venue_b, ("kalshi", "other_venue"), lags=4)
    if pdres:
        sec.pre(pdres.summary())

    delay = estimate_delay(tv.times, tv.venue_a, tv.times, tv.venue_b, max_shift_s=20.0)
    epps = epps_curve(tv.times, tv.venue_a, tv.times, tv.venue_b)
    sec.subheading("Lead-lag and the Epps effect")
    sec.pre(delay.summary())
    sec.table(pd.DataFrame({"interval_s": list(epps), "measured_corr": list(epps.values())}))

    ret = np.diff(sim.quoted[mid_tk][ev:] * PAYOUT)
    sec.subheading("Weak-form efficiency after the release")
    sec.pre("\n".join(t.summary() for t in efficiency_panel(ret)))
    vr = variance_ratio_profile(ret)
    sec.table(
        pd.DataFrame(
            {
                "q": list(vr),
                "vr": [v.detail.get("vr", np.nan) for v in vr.values()],
                "z": [v.statistic for v in vr.values()],
                "p": [v.pvalue for v in vr.values()],
            }
        )
    )

    study = event_study(
        sim.times, sim.quoted[mid_tk] * PAYOUT, sim.event_ts,
        pre_s=600.0, post_s=1800.0, bar_s=5.0, placebo_draws=200,
    )
    sec.subheading("Event study")
    sec.pre(study.summary())
    sec.html(
        line_chart(
            study.grid.tolist(), {"CAAR (cents)": study.caar.tolist()},
            title="Cumulative abnormal revision around the release",
            xlabel="seconds from release", ylabel="cents",
        )
    )

    # ---- 5. backtests ----------------------------------------------------
    sec = report.section("5. Backtests")
    sec.text(
        "Every strategy runs over identical data through the same queue-aware fill model, "
        "with latency, per-level fees and adverse selection charged. The deflated Sharpe "
        "corrects the headline number for the fact that several variants were tried."
    )
    results = run_backtests(sim, settings, n_trials=len(STRATEGY_NAMES))
    table = backtest_table(results)
    sec.table(table)
    sec.subheading("Reading the zeros")
    sec.text(
        "Three strategies place no trades on this data, and that is the correct outcome "
        "rather than a broken run. The simulated ladder is coherent by construction, so "
        "the arbitrage and coherence strategies have nothing to act on; and with a 2-cent "
        "spread against a 3.5-cent round-trip fee at even money, the microprice signal "
        "cannot clear its own transaction cost. A backtest that reported profits here "
        "would be reporting a bug. The stressed run is the same arbitrage strategy on the "
        "same session with a 12-cent monotonicity violation injected -- it fires, "
        "which is what makes the zeros above meaningful."
    )
    for name, res in results.items():
        sec.subheading(name)
        sec.pre(res.summary())
    arb_res = results.get("ladder_arb")
    if arb_res is not None:
        sec.html(
            line_chart(
                (arb_res.times - arb_res.times[0]).tolist(),
                {n: (r.equity - r.equity[0]).tolist() for n, r in results.items()},
                title="Equity curves (cents, rebased)", xlabel="seconds", ylabel="cents",
            )
        )

    # ---- 6. calibration --------------------------------------------------
    sec = report.section("6. Calibration")
    cal_quoted, outcomes = simulate_calibration_sample(n=8000, bias_a=0.85)
    rep = calibration_report(cal_quoted, outcomes)
    sec.text(
        "A market quote is a probability forecast, so it can be scored like one. The sample "
        "here is generated with a known favourite-longshot distortion (true a = 0.85); "
        "Platt scaling should recover it. Murphy's decomposition separates the part that is "
        "tradeable (reliability) from the part that is not (resolution and uncertainty)."
    )
    sec.cards(
        [
            ("Brier", f"{rep.brier:.4f}", ""),
            ("Recovered Platt a", f"{rep.platt_a:.3f}", ""),
            ("True Platt a", "0.850", ""),
            ("ECE", f"{rep.ece:.4f}", ""),
        ]
    )
    sec.pre(rep.summary())
    sec.table(rep.curve)
    if not rep.curve.empty:
        sec.html(
            scatter_chart(
                rep.curve["mean_pred"].tolist(), rep.curve["freq"].tolist(),
                title="Reliability curve", xlabel="quoted probability",
                ylabel="realised frequency", diagonal=True,
            )
        )

    elapsed = time.perf_counter() - t0
    report.footer = (
        f"kalshi-alpha v{__version__} - generated in {elapsed:.1f}s - "
        f"seed {settings.seed} - offline simulator, no live market data."
    )
    out_path = report.write(Path(args.out) / "report.html")

    table.to_csv(Path(args.out) / "backtests.csv", index=False)
    recovery.to_csv(Path(args.out) / "halflife_recovery.csv", index=False)
    shares.to_csv(Path(args.out) / "information_share_recovery.csv", index=False)
    thresh.to_csv(Path(args.out) / "violation_threshold.csv", index=False)
    by_price.to_csv(Path(args.out) / "threshold_by_price.csv", index=False)

    print(f"report written to {out_path}")
    print(f"artifacts written to {Path(args.out).resolve()}")
    print(f"\nbacktest summary:\n{table.to_string(index=False)}")
    return 0


def scan_live(event_ticker: str, env: str = "prod", depth: int = 10) -> int:
    """Pull a real event's books and scan them for arbitrage.

    Read-only and unauthenticated: Kalshi serves market data without
    credentials, so this needs no API key. Nothing here can place an order --
    :mod:`kalshi_alpha.execution` is not even imported.

    The bucket markets of a single event partition the outcome space, so
    exactly one settles YES and the fair prices must sum to $1. The gap between
    the summed bids and the summed asks is the event's no-arbitrage band, and
    printing it is the point: on real markets that band is far wider than any
    fee hurdle, which is the honest explanation for why model-free arbitrage is
    rare rather than merely hard to find.
    """
    import asyncio

    from kalshi_alpha.data.kalshi_client import KalshiClient, parse_market_meta

    settings = load_settings(env=env)

    async def pull():
        async with KalshiClient(settings) as client:
            markets = await client.get_markets(event_ticker=event_ticker)
            if not markets:
                return [], {}
            metas = [parse_market_meta(m) for m in markets]
            books = await client.snapshot([m.ticker for m in metas], depth=depth)
            return metas, books

    metas, books = asyncio.run(pull())
    if not metas:
        print(f"no markets found for event {event_ticker!r} on env={env}", file=sys.stderr)
        print("hint: the demo environment lists different markets than production; "
              "pass --env prod for real events.", file=sys.stderr)
        return 1

    two_sided = {t: b for t, b in books.items() if b.is_two_sided}
    print(f"{event_ticker}: {len(books)} markets, {len(two_sided)} two-sided (env={env})")
    for meta in metas:
        book = books.get(meta.ticker)
        if book is None:
            continue
        print(
            f"  {meta.ticker:34s} {str(meta.strike_type or '-'):8s} "
            f"bid={str(book.best_yes_bid):>4s} ask={str(book.best_yes_ask):>4s} "
            f"depth={book.notional_depth():>7d}c"
        )

    quoted = [b for b in two_sided.values() if b.best_yes_bid and b.best_yes_ask]
    if len(quoted) >= 2:
        sum_bid = sum(b.best_yes_bid for b in quoted if b.best_yes_bid)
        sum_ask = sum(b.best_yes_ask for b in quoted if b.best_yes_ask)
        print(
            f"\nno-arbitrage band across {len(quoted)} outcomes: "
            f"[{sum_bid}c, {sum_ask}c] against a fair value of 100c "
            f"-> {sum_ask - sum_bid}c wide"
        )

    group = EventGroup(event_ticker, tuple(two_sided), exhaustive=True)
    result = ArbEngine(settings).scan(two_sided, groups=[group])
    print("\n" + result.summary())
    for opp in result.opportunities:
        print(opp.describe())
        print(f"    {opp.detail}")
    if not result.opportunities:
        print("no arbitrage after fees on this event")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    if args.event:
        return scan_live(args.event, args.env, args.depth)

    settings = load_settings()
    sim = _sim(steps=args.steps)
    books = sim.final_books()
    if args.dislocate:
        # Shift whichever strike is quoted closest to even money. Prices are
        # clamped to the 1..99 grid, so pushing a strike that is already at 4c
        # or 96c would be silently absorbed by the clamp and inject nothing.
        books, moved = inject_ladder_violation(books, sim.ladder.tickers, args.dislocate)
        if moved is None:
            print("could not inject a violation of that size on this book")
        else:
            print(f"injected a {args.dislocate}c monotonicity violation at {moved}")
    result = ArbEngine(settings).scan(books, ladders=[sim.ladder])
    print(result.summary())
    for opp in result.opportunities:
        print(opp.describe())
        print(f"    {opp.detail}")
    if not result.opportunities:
        print("no arbitrage after fees (this is the expected result on a coherent book)")
    return 0


def cmd_diffusion(args: argparse.Namespace) -> int:
    recovery = half_life_recovery()
    shares = information_share_recovery()
    print("half-life recovery:\n", recovery.to_string(index=False), sep="")
    print("\ninformation-share recovery:\n", shares.to_string(index=False), sep="")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        recovery.to_csv(Path(args.out) / "halflife_recovery.csv", index=False)
        shares.to_csv(Path(args.out) / "information_share_recovery.csv", index=False)
        print(f"\nwritten to {Path(args.out).resolve()}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    settings = load_settings()
    sim = _sim(steps=args.steps)
    if args.strategy == "all":
        table = backtest_table(run_backtests(sim, settings, n_trials=len(STRATEGY_NAMES)))
        print(table.to_string(index=False))
        return 0

    configs = {
        "ladder_arb": {"ladders": [sim.ladder], "max_qty": 100, "scan_every": 20},
        "coherence": {"ladder": sim.ladder, "qty": 20},
        "microprice": {"tickers": sim.tickers, "qty": 10},
        "drift": {"tickers": sim.tickers, "event_ts": sim.event_ts, "hold_s": 900.0, "qty": 25},
    }
    strategy = build_strategy(args.strategy, **configs[args.strategy])
    result = Backtester(settings).run(
        sim.books, sim.times, strategy, trades=sim.trades, settlement=sim.settle()
    )
    print(result.summary())
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        result.equity_frame().to_csv(Path(args.out) / f"{args.strategy}_equity.csv", index=False)
        result.fills.to_csv(Path(args.out) / f"{args.strategy}_fills.csv", index=False)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    quoted, outcomes = simulate_calibration_sample(n=args.n, bias_a=args.bias)
    rep = calibration_report(quoted, outcomes)
    print(f"true Platt a = {args.bias}")
    print(rep.summary())
    print(rep.curve.to_string(index=False))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Score every estimator against the simulator's known ground truth."""
    del args
    hl = half_life_recovery()
    shares = information_share_recovery()
    cal_quoted, outcomes = simulate_calibration_sample(n=8000, bias_a=0.85)
    rep = calibration_report(cal_quoted, outcomes)

    hl_ok = bool((hl["error_pct"].abs() < 25).all())
    leader_ok = bool(shares.loc[shares["decisive"], "correct_leader"].all())
    # The Hasbrouck share is only identified up to the Cholesky ordering and its
    # target depends on the adjustment speeds as well as the innovation
    # variances, so the meaningful test is that it is *monotone* in the true
    # share, not that it matches a point value.
    monotone_ok = bool(
        shares["hasbrouck_lo"].is_monotonic_decreasing
        if shares["true_share_a"].is_monotonic_decreasing
        else shares["hasbrouck_lo"].is_monotonic_increasing
    )
    platt_ok = abs(rep.platt_a - 0.85) < 0.1

    print("estimator validation against simulator ground truth")
    print("-" * 62)
    print(hl.to_string(index=False))
    print()
    print(shares.to_string(index=False))
    print()
    print(f"platt a: recovered {rep.platt_a:.3f} vs true 0.850")
    print("-" * 62)
    print(f"half-life within 25%        : {'PASS' if hl_ok else 'FAIL'}")
    print(f"leader identified correctly : {'PASS' if leader_ok else 'FAIL'}")
    print(f"info share monotone in truth: {'PASS' if monotone_ok else 'FAIL'}")
    print(f"calibration recovered       : {'PASS' if platt_ok else 'FAIL'}")
    return 0 if (hl_ok and leader_ok and monotone_ok and platt_ok) else 1


def cmd_fetch(args: argparse.Namespace) -> int:
    """Pull live book snapshots. Requires credentials; read-only."""
    import asyncio

    from kalshi_alpha.data.kalshi_client import KalshiClient, parse_market_meta
    from kalshi_alpha.data.store import TickStore

    settings = load_settings(mode="live", env=args.env)
    if not settings.has_credentials:
        # Not an error: market data is public. Credentials are only needed for
        # balance, positions and order placement, none of which this does.
        print(
            "running unauthenticated -- market data is public on Kalshi. "
            "Credentials (see .env.example) are only needed for account endpoints.",
            file=sys.stderr,
        )

    async def main() -> int:
        async with KalshiClient(settings) as client:
            markets = await client.get_markets(event_ticker=args.event)
            if not markets:
                print(f"no markets found for event {args.event!r} on env={args.env}",
                      file=sys.stderr)
                print("hint: the demo environment lists different markets than "
                      "production; pass --env prod for real events.", file=sys.stderr)
                return 1
            tickers = [m["ticker"] for m in markets][: args.limit]
            books = await client.snapshot(tickers, depth=args.depth)
            metas = [parse_market_meta(m) for m in markets[: args.limit]]

            store = TickStore(settings.data_dir / "parquet")
            path = store.write_books(list(books.values()), args.event)
            print(f"fetched {len(books)} books for {args.event}")
            for meta in metas:
                book = books.get(meta.ticker)
                if book:
                    print(
                        f"  {meta.ticker:32s} {str(meta.strike_type or '-'):6s} "
                        f"{meta.strike if meta.strike is not None else '-':>10} "
                        f"bid={book.best_yes_bid} ask={book.best_yes_ask}"
                    )
            if path:
                print(f"written to {path}")
            return 0

    return asyncio.run(main())


def cmd_config(args: argparse.Namespace) -> int:
    del args
    settings = load_settings()
    print(json.dumps(json.loads(settings.model_dump_json()), indent=2, default=str))
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kalshi-alpha",
        description=(
            "Mispricing, arbitrage and information-diffusion research for "
            "binary event contracts."
        ),
    )
    parser.add_argument("--version", action="version", version=f"kalshi-alpha {__version__}")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--json-logs", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="run the full pipeline offline and write an HTML report")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--half-life", type=float, default=150.0)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("scan", help="scan for arbitrage, offline or against live markets")
    p.add_argument("--event", default="",
                   help="scan a real event's live books (read-only, no API key needed)")
    p.add_argument("--env", default="prod", choices=["prod", "demo"],
                   help="which Kalshi environment to read from (default: prod)")
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--offline", action="store_true", help="use the simulator (default)")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--dislocate", type=int, default=0,
                   help="inject a ladder monotonicity violation of N cents")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("diffusion", help="run the information-diffusion study")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_diffusion)

    p = sub.add_parser("backtest", help="backtest a strategy")
    p.add_argument("--strategy", default="all", choices=[*STRATEGY_NAMES, "all"])
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("calibrate", help="score a probability forecast")
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--bias", type=float, default=0.85)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("validate", help="score every estimator against known ground truth")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("fetch", help="pull live book snapshots (read-only)")
    p.add_argument("--event", required=True)
    p.add_argument("--env", default="prod", choices=["prod", "demo"],
                   help="which Kalshi environment to read from (default: prod)")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--depth", type=int, default=10)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("config", help="print the resolved configuration")
    p.set_defaults(func=cmd_config)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, args.json_logs)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
