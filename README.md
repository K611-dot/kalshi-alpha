# kalshi-alpha

**Mispricing detection, no-arbitrage enforcement, and information-diffusion measurement for binary event contracts.**

Two questions about prediction markets, answered with the same data layer:

1. **Is the quoted surface internally consistent, and if not, is the inconsistency big enough to trade after fees?**
2. **How fast does new information actually get into the price after a scheduled release?**

Everything runs offline against a simulator whose ground truth is known by construction, so every estimator in the repo is *scored against a known answer* rather than merely executed. One command reproduces every number below.

```bash
pip install -e ".[dev,plots]"
make demo        # writes artifacts/report.html
make validate    # scores every estimator against ground truth
```

---

## Why a simulator is the point, not a shortcut

On live data none of these quantities are observable. You cannot check whether an estimated half-life of 90 seconds is right, because the true half-life is not a thing you can look up — so a subtly wrong estimator is indistinguishable from a correct one, forever.

So the package ships a generator where the answer is known:

- a latent underlying follows a driftless GBM and each market's fair probability is the closed-form digital price `Φ(d₂)`, which makes the whole strike ladder **coherent and monotone by construction** — the arbitrage scanner must find nothing, and when a violation is deliberately injected it must find exactly that one;
- a scheduled release jumps the underlying and the quote converges to its new fair value with a **specified** half-life;
- two venues each originate their own information shocks, giving the Hasbrouck and Gonzalo–Granger shares a known target.

`make validate` scores the estimators against those targets. That is what turns "the code runs" into "the method is correct."

### Estimator recovery (`kalshi-alpha validate`)

| true half-life | estimated | error | identified |
|---:|---:|---:|:--|
| 30 s | 29.3 s | −2.4% | yes |
| 60 s | 62.0 s | +3.3% | yes |
| 120 s | 122.8 s | +2.4% | yes |
| 300 s | 306.4 s | +2.1% | yes |
| 600 s | 594.8 s | −0.9% | yes |

| true info share (venue A) | Hasbrouck bounds | leader identified |
|---:|:--|:--|
| 0.90 | [0.954, 0.958] | correct |
| 0.66 | [0.776, 0.781] | correct |
| 0.50 | [0.606, 0.613] | tie — not graded |
| 0.10 | [0.130, 0.133] | correct |

Calibration: injected Platt distortion `a = 0.850`, recovered `0.883`.

---

## Headline result: the fee curve decides where arbitrage exists

Kalshi's taker fee is `ceil(0.07 · C · P · (1 − P))` — **concave, peaking at even money**. A round trip costs 3.5c at 50c and 0.67c at 5c.

That single fact determines where model-free edge is capturable. Build a ladder pair at a given price level and widen the monotonicity violation one cent at a time until it becomes tradeable:

| price level | min tradeable violation | predicted hurdle `2·r·p·(1−p)` |
|---:|---:|---:|
| 5c | **1c** | 0.67c |
| 20c | 3c | 2.24c |
| 50c | **4c** | 3.50c |
| 80c | 3c | 2.24c |
| 90c | 2c | 1.26c |

The empirical threshold matches the closed-form prediction, rounded up to the one-cent grid. **A violation that is already tradeable at 5c needs to be four times larger at even money.** That is structural, not market-specific, and it says exactly where to point a scanner.

---

## What is in here

### Arbitrage — five structures, all proved before they are reported

| structure | violated condition |
|---|---|
| `crossed_book` | `bid_YES < ask_YES` within one market |
| `dutch_book_under` | `Σ ask ≥ $1` over an exhaustive group |
| `dutch_book_over` | `Σ bid ≤ $1` over an exclusive group |
| `ladder_monotonicity` | `P(X ≥ k)` non-increasing in `k` |
| `cross_venue` | law of one price across exchanges |

Plus a **general LP** that drops the names entirely: every displayed price level becomes a tradeable instrument, and the programme maximises the worst-case settlement P&L subject to displayed depth and a capital budget. `z* > 0` is a certificate of arbitrage. Because two levels with identical payoff differ only in cost, the LP walks the book by itself.

Its **dual** is just as useful. By the fundamental theorem of asset pricing, no arbitrage exists iff a risk-neutral measure consistent with every quote exists — so when the scanner finds nothing it returns that measure instead, giving per-market probabilities that are mutually consistent (they sum to one across an exhaustive group and respect ladder monotonicity) rather than read off each mid independently. Maximising and minimising an arbitrary claim's price over that set yields **model-free super- and sub-replication bounds** for contracts the market has not quoted tightly.

Every reported opportunity is validated state by state: it is only an arbitrage if the **worst** settlement state is strictly profitable. A property-based test asserts exactly this over randomly generated books.

### Information diffusion

- **Price discovery** — VECM with the cointegrating vector fixed at `(1, −1)`, then Hasbrouck information shares with **exact Cholesky-ordering bounds** (never a single number — the bounds are widest precisely when both venues are fast, which is when the question matters most) and Gonzalo–Granger component shares. ADF on the spread licenses the interpretation.
- **Half-life of adjustment** — three estimators that fail differently: non-parametric adjustment profile, weighted exponential decay, and AR(1) on the gap. When a leading series is available, the **spread-based** estimator wins outright, because it is the only one that does not assume the efficient price stops moving after the release.
- **Window selection** — the single-series estimators are only identified when the half-life is short relative to the horizon over which residual volatility is stable. `window_scan` exposes the plateau-then-runaway curve, and `select_window` locates the knee. When no plateau exists the result is returned as **unidentified** rather than dressed up as an answer.
- **Lead–lag** — Hayashi–Yoshida on asynchronous data, avoiding the Epps effect that makes grid-sampled correlation collapse at exactly the horizon you care about. Scanning the clock shift traces out the propagation delay.
- **Event study** — event-relative alignment, drift-adjusted CAAR, cross-sectional t-stats, and a **placebo distribution** from randomly-timed pseudo-events, because a parametric t-stat on eight fat-tailed events is not credible.
- **Efficiency tests** — Lo–MacKinlay variance ratio (heteroskedasticity-robust by default; the homoskedastic version rejects everywhere on event contracts because volatility collapses at the tails), Wald–Wolfowitz runs, Ljung–Box.

### Microstructure

Order-flow imbalance (Cont–Kukanov–Stoikov), Stoikov microprice, Kyle's lambda with White standard errors, Amihud illiquidity, Roll's implied spread, VPIN on a volume clock, and the effective/realised/impact spread decomposition. Strict sequence-checked L2 book reconstruction that **raises on a gap** rather than quoting against a book that has diverged.

### Backtesting

Event-driven, with the things that usually get skipped:

- **queue position** — a passive order joins the back of the queue and only fills after the size ahead is consumed;
- **latency** — orders trade against the book at arrival, not the book the signal was computed on;
- **per-level fees** — an order sweeping three levels pays three separately-rounded fees;
- **adverse selection** — orders the market trades through are filled at the worst moment;
- **queue leakage**, **marketable-limit detection**, **explicit settlement**.

Performance is reported with the corrections that expose overfitting: **deflated Sharpe** (Bailey–López de Prado, adjusting for the number of variants tried, return skew, and kurtosis), probabilistic Sharpe, and a **stationary bootstrap** confidence band that preserves serial dependence.

### Execution

RSA-PSS-signed async REST client with a token-bucket rate limiter and retries on idempotent verbs only; sequence-checked websocket that re-subscribes on a gap; an OMS with an explicit state machine that raises on illegal transitions, idempotent client order IDs, atomic multi-leg submission with unwind, and exchange reconciliation. A **latching** kill switch — one that re-arms itself has not stopped the loss, it has scheduled it.

---

## Reading the backtest zeros

```
strategy               pnl_usd  trades  fees_usd   sharpe  deflated_sr
ladder_arb                0.00       0      0.00      NaN          NaN
coherence                 0.00       0      0.00      NaN          NaN
microprice                0.00       0      0.00      NaN          NaN
drift                    19.15       7      2.35   102.44        0.743
ladder_arb_stressed     386.70      87     95.30    20.13        0.227
```

Three strategies place no trades, and that is the correct outcome. The simulated ladder is coherent by construction, so the arbitrage and coherence strategies have nothing to act on; and with a 2-cent spread against a 3.5-cent round-trip fee, the microprice signal cannot clear its own transaction cost. **A backtest reporting profits here would be reporting a bug.**

`ladder_arb_stressed` is the same strategy on the same session with a 12-cent violation injected. It fires, and fees eat 20% of gross. That pair — silent on clean data, capturing on dirty data — is the result; either half alone proves nothing.

The `drift` Sharpe of 102 is an artefact of annualising a 5-second bar over a single session and should be read as "it worked on one event", which is what the deflated Sharpe of 0.74 is there to say.

---

## Commands

```bash
kalshi-alpha demo                      # full pipeline -> artifacts/report.html
kalshi-alpha validate                  # score estimators against ground truth
kalshi-alpha scan --dislocate 12       # inject a 12c violation and detect it
kalshi-alpha diffusion --out artifacts
kalshi-alpha backtest --strategy all
kalshi-alpha calibrate --bias 0.85
kalshi-alpha scan --event KXGDPYEAR-36  # scan REAL live markets (no API key needed)
kalshi-alpha fetch --event KXGDPYEAR-36 # pull live snapshots to parquet
kalshi-alpha config
```

**No API key is required.** Kalshi serves market data publicly, so `scan --event` and `fetch` work unauthenticated. Credentials are only needed for account endpoints (balance, positions) and order placement — and the client refuses to place an order without them. The package defaults to `offline` mode, so importing it can never reach the network.

### Live scan against a real event

```
KXGDPYEAR-36: 14 markets, 14 two-sided (env=prod)
  KXGDPYEAR-36-T6.0      gt       bid=  6 ask=  7
  KXGDPYEAR-36-B5.3      between  bid=  1 ask=  2
  ...
no-arbitrage band across 14 outcomes: [80c, 123c] against a fair value of 100c -> 43c wide
found 0 opportunities
```

The bucket markets of an event partition the outcome space, so the fair prices must sum to $1. On this real event you would pay **$1.23** for a guaranteed $1, or receive **$0.80** to take on a $1 liability. That 43-cent band — against a ~3.5c fee hurdle — is the honest reason model-free arbitrage on this exchange is rare: not that it is hard to detect, but that quoted spreads are an order of magnitude wider than any dislocation worth chasing.

---

## Layout

```
src/kalshi_alpha/
├── types.py             domain model: books, legs, positions, settlement
├── config.py            typed settings: fees, risk, execution, estimators
├── arbitrage/           fees, liquidity, payoff proof, detectors, LP + dual
├── diffusion/           VECM/Hasbrouck, half-life, lead-lag, event study, efficiency
├── microstructure/      L2 reconstruction, OFI/microprice, Kyle/Amihud/VPIN/Roll
├── probability/         Brier/Murphy, isotonic + Platt, no-arbitrage projection
├── backtest/            queue-aware fills, event loop, deflated Sharpe, strategies
├── execution/           OMS state machine, pre-trade risk, paper broker
├── data/                Kalshi REST/WS client, parquet store, calendar, simulator
└── report/              self-contained HTML
```

Docs: [METHODOLOGY](docs/METHODOLOGY.md) (the maths and why each choice was made) · [ARCHITECTURE](docs/ARCHITECTURE.md) · [RESULTS](docs/RESULTS.md)

---

## Tests

298 tests, `ruff`-clean. The ones that matter:

- **`test_recovery.py`** — estimators scored against simulator ground truth.
- **`test_properties.py`** — hypothesis property tests stating the no-arbitrage theorems directly: *every* reported arbitrage is profitable in *every* settlement state; a normal book is never a Dutch book against itself; a hedged pair pays identically in both states.
- **`test_arbitrage.py`** — the false-negative tests matter more than the true positives. A scanner that reports edge which is not there loses money on every signal.

```bash
make test      # pytest
make lint      # ruff
make cov       # coverage
```

## License

MIT
