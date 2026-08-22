"""Generate standalone chart images from real pipeline output.

Every figure here is computed by the package, not transcribed into the script.
That is the point: a chart in a README or a talk is a claim, and a claim you
cannot regenerate is one you cannot defend when someone asks how you got it.

    python scripts/make_figures.py --out artifacts/figures

Writes PNGs at presentation resolution, styled to be legible at thumbnail size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kalshi_alpha.arbitrage.fees import round_trip_breakeven_cents
from kalshi_alpha.config import Settings
from kalshi_alpha.data.synthetic import SimConfig, simulate_calibration_sample, simulate_ladder
from kalshi_alpha.diffusion.event_study import event_study
from kalshi_alpha.diffusion.halflife import window_scan
from kalshi_alpha.probability.calibration import calibration_report
from kalshi_alpha.types import PAYOUT

# Palette shared with the rest of the project's presentation material.
INK = "#EDF1F5"
MUTE = "#93A1B0"
DIM = "#64717F"
CARD = "#131A24"
INSET = "#1B2430"
RULE = "#2C3846"
BLUE = "#7B94FF"
AMBER = "#E2A33C"
ROSE = "#E0637A"
MINT = "#5FCFA8"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": CARD,
            "axes.facecolor": CARD,
            "savefig.facecolor": CARD,
            "text.color": INK,
            "axes.labelcolor": MUTE,
            "axes.edgecolor": RULE,
            "xtick.color": DIM,
            "ytick.color": DIM,
            "grid.color": RULE,
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.titleweight": "semibold",
            "axes.labelsize": 13,
            "legend.frameon": False,
            "legend.labelcolor": MUTE,
            "figure.dpi": 160,
        }
    )


def _finish(ax, title: str, subtitle: str = "") -> None:
    """Title above subtitle above axes, with enough pad that they never collide.

    Matplotlib places a title a fixed number of points above the axes, so the
    subtitle has to be given that space explicitly -- otherwise both land in
    the same strip and overprint each other.
    """
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, pad=38 if subtitle else 14, loc="left")
    if subtitle:
        ax.text(
            0.0, 1.018, subtitle, transform=ax.transAxes, color=DIM,
            fontsize=11.5, va="bottom", ha="left",
        )


def _save(fig, out: Path, name: str) -> Path:
    path = out / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# --------------------------------------------------------------------------
def fig_calibration(out: Path) -> None:
    """Reliability curve: the honest 'model evaluation' chart in this project.

    A market quote is a probability forecast, so it is scored exactly as any
    probabilistic classifier would be. The sample carries a known
    favourite-longshot distortion, and Platt scaling recovers it.
    """
    quoted, outcomes = simulate_calibration_sample(n=20_000, bias_a=0.85, seed=4)
    rep = calibration_report(quoted, outcomes, n_bins=12)
    curve = rep.curve

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13.6, 5.8), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    fig.subplots_adjust(wspace=0.42)

    ax.plot([0, 1], [0, 1], "--", color=DIM, lw=1.6, label="perfect calibration", zorder=1)
    ax.errorbar(
        curve["mean_pred"], curve["freq"],
        yerr=[curve["freq"] - curve["lo"], curve["hi"] - curve["freq"]],
        fmt="o", color=BLUE, ecolor=RULE, elinewidth=2.2, capsize=0,
        markersize=9, label="quoted vs realised", zorder=3,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("quoted probability")
    ax.set_ylabel("realised frequency")
    ax.legend(loc="upper left")
    _finish(ax, "Is the market calibrated?",
            "when it says 30%, does it happen 30% of the time?")

    # Murphy decomposition: only reliability is tradeable.
    d = rep.decomposition
    parts = ["reliability\n(miscalibration)", "resolution\n(discrimination)",
             "uncertainty\n(irreducible)"]
    vals = [d.reliability, d.resolution, d.uncertainty]
    colors = [ROSE, MINT, DIM]
    bars = ax2.barh(parts, vals, color=colors, height=0.55)
    for bar, v in zip(bars, vals, strict=True):
        ax2.text(v + d.uncertainty * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}", va="center", color=INK, fontsize=12.5)
    ax2.set_xlim(0, d.uncertainty * 1.38)
    ax2.invert_yaxis()
    ax2.tick_params(labelsize=11.5)
    _finish(ax2, "Where the score comes from",
            "only the top bar is tradeable")
    ax2.grid(axis="y", alpha=0)

    fig.text(
        0.5, -0.04,
        f"Brier {rep.brier:.4f}    ECE {rep.ece:.4f}    "
        f"Platt a = {rep.platt_a:.3f} against an injected 0.850    |    "
        f"miscalibration is {d.uncertainty / max(d.reliability, 1e-9):.0f}x smaller "
        f"than the irreducible variance",
        ha="center", color=MUTE, fontsize=12,
    )
    _save(fig, out, "calibration.png")


def fig_window_scan(out: Path) -> None:
    """The methodological point: the estimate depends on how much data you feed it."""
    cfg = SimConfig(n_steps=1600, dt_s=5.0, event_step=600,
                    adjustment_half_life_s=120.0, event_jump_cents=18.0)
    sim = simulate_ladder(cfg=cfg)
    tk = sim.tickers[len(sim.tickers) // 2]
    ev = 600
    scan = window_scan(sim.times[ev:] - sim.times[ev], sim.quoted[tk][ev:] * PAYOUT, 5.0)
    scan = scan.dropna(subset=["half_life_s"])

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.plot(scan["seconds"], scan["half_life_s"], "-o", color=BLUE, lw=2.6, markersize=8)
    ax.axhline(120.0, color=AMBER, ls="--", lw=2, label="true half-life = 120 s")

    # Everything below is positioned from the data's own range, so the figure
    # survives a different seed or a different simulated half-life rather than
    # silently placing its annotations off-axis.
    top = float(scan["half_life_s"].max())
    lo_s = float(scan["seconds"].min())
    ax.set_ylim(0, top * 1.20)

    plateau = scan[scan["seconds"] <= 760]
    ax.axvspan(lo_s, 760, color=MINT, alpha=0.10)
    ax.text(
        float(np.sqrt(max(plateau["seconds"].min(), lo_s) * 760)), top * 0.55,
        "plateau\n(the answer)", color=MINT, fontsize=13.5, ha="center", va="center",
    )

    worst = scan.iloc[-1]
    ax.annotate(
        "runaway: the efficient price resumes being a\nrandom walk and swamps the decay",
        xy=(float(worst["seconds"]), float(worst["half_life_s"])),
        xytext=(lo_s * 3.4, top * 0.94),
        color=ROSE, fontsize=12.5, va="center",
        arrowprops={"arrowstyle": "->", "color": ROSE, "lw": 1.8,
                    "connectionstyle": "arc3,rad=-0.18"},
    )

    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks([200, 400, 800, 1600, 3200])
    ax.set_xlabel("estimation window (seconds, log)")
    ax.set_ylabel("estimated half-life (seconds)")
    ax.legend(loc="upper left")
    _finish(ax, "Why you cannot just pick a window",
            "same data, same estimator - only the amount of data changes")
    fig.text(0.5, -0.02,
             "Reporting one number without this curve is how a two-minute "
             "half-life gets published as forty.",
             ha="center", color=MUTE, fontsize=12.5)
    _save(fig, out, "window_scan.png")


def fig_fee_threshold(out: Path) -> None:
    """Measured minimum tradeable violation against the closed-form hurdle."""
    from kalshi_alpha.cli import threshold_by_price

    df = threshold_by_price(Settings())
    grid = np.arange(1, 100)
    hurdle = [round_trip_breakeven_cents(int(p)) for p in grid]

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.bar(df["price_cents"], df["min_violation_cents"], width=6.0,
           color=BLUE, alpha=0.9, label="measured minimum (1c grid)")
    ax.plot(grid, hurdle, color=AMBER, lw=3,
            label=r"predicted hurdle  $2 r p (1-p)$")

    # Both labels sit in the empty band above the tallest bar, so neither
    # overprints the data it is pointing at.
    ax.annotate("1c is already\ntradeable here", xy=(5, 1.06), xytext=(12, 4.55),
                color=MINT, fontsize=12.5, ha="left", va="center",
                arrowprops={"arrowstyle": "->", "color": MINT, "lw": 1.6,
                            "connectionstyle": "arc3,rad=0.18"})
    ax.annotate("4x larger needed\nat even money", xy=(53, 4.06), xytext=(63, 4.85),
                color=ROSE, fontsize=12.5, ha="left", va="center",
                arrowprops={"arrowstyle": "->", "color": ROSE, "lw": 1.6,
                            "connectionstyle": "arc3,rad=-0.25"})

    ax.set_xlabel("contract price (cents)")
    ax.set_ylabel("minimum tradeable violation (cents)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 5.4)
    ax.legend(loc="upper left", fontsize=12)
    _finish(ax, "The fee curve decides where arbitrage exists",
            "measurement lands exactly on the closed form, rounded to the cent grid")
    _save(fig, out, "fee_threshold.png")


def fig_halflife_recovery(out: Path) -> None:
    """Estimator scored against half-lives it was never told."""
    from kalshi_alpha.cli import half_life_recovery

    df = half_life_recovery()
    fig, ax = plt.subplots(figsize=(8.4, 7.4))

    lo, hi = 20, 900
    ax.plot([lo, hi], [lo, hi], "--", color=AMBER, lw=2, label="perfect recovery")
    ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1],
                    color=AMBER, alpha=0.08, label="+/- 10%")
    ax.scatter(df["true_half_life_s"], df["estimated_s"], s=160, color=BLUE,
               zorder=5, label="measured", edgecolor=CARD, linewidth=2)

    for _, row in df.iterrows():
        ax.annotate(f"{row['error_pct']:+.1f}%",
                    (row["true_half_life_s"], row["estimated_s"]),
                    textcoords="offset points", xytext=(14, -14),
                    color=MUTE, fontsize=12)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("true half-life (seconds)")
    ax.set_ylabel("estimated half-life (seconds)")
    ticks = [30, 60, 120, 300, 600]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])
    ax.legend(loc="upper left")
    _finish(ax, "Scored against a known answer",
            "the simulator's true half-life is never shown to the estimator")
    _save(fig, out, "halflife_recovery.png")


def fig_event_study(out: Path) -> None:
    """Cumulative abnormal repricing around a scheduled release."""
    cfg = SimConfig(n_steps=2400, dt_s=5.0, event_step=900,
                    adjustment_half_life_s=180.0, event_jump_cents=18.0, seed=11)
    sim = simulate_ladder(cfg=cfg)
    tk = sim.tickers[len(sim.tickers) // 2]
    res = event_study(sim.times, sim.quoted[tk] * PAYOUT, sim.event_ts,
                      pre_s=600.0, post_s=1800.0, bar_s=5.0, placebo_draws=150)

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.axvline(0, color=ROSE, lw=2, label="release")
    ax.axhline(0, color=RULE, lw=1)
    ax.fill_between(res.grid, res.caar - 1.96 * res.stderr, res.caar + 1.96 * res.stderr,
                    color=BLUE, alpha=0.16)
    ax.plot(res.grid, res.caar, color=BLUE, lw=2.8, label="cumulative abnormal repricing")

    ax.set_xlabel("seconds from release")
    ax.set_ylabel("cents")
    ax.legend(loc="best")
    _finish(ax, "How news gets into the price",
            "aligned on the release, pre-event drift removed, placebo-tested")
    fig.text(0.5, -0.02,
             "The jump is not the trade. The drift after it is - and only if it "
             "outruns the fee hurdle.",
             ha="center", color=MUTE, fontsize=12.5)
    _save(fig, out, "event_study.png")


FIGURES = {
    "calibration": fig_calibration,
    "window_scan": fig_window_scan,
    "fee_threshold": fig_fee_threshold,
    "halflife_recovery": fig_halflife_recovery,
    "event_study": fig_event_study,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/figures")
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()

    chosen = args.only or list(FIGURES)
    print(f"generating {len(chosen)} figures into {out.resolve()}")
    for name in chosen:
        FIGURES[name](out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
