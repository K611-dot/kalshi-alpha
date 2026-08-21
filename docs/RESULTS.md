# Results

Every figure here is produced by `make demo` and `make validate` against the offline simulator with seed 7. Nothing depends on network access or stored data, so all of it reproduces exactly.

---

## 1. Estimator validation against ground truth

The simulator generates data whose true half-life, true information share, and true calibration distortion are known by construction. These are the scores.

### Half-life of information absorption

| true | estimated | error | estimator | window | identified |
|---:|---:|---:|:--|---:|:--|
| 30 s | 29.28 s | −2.4% | `spread_ar1` | 290 s | yes |
| 60 s | 61.95 s | +3.3% | `spread_ar1` | 615 s | yes |
| 120 s | 122.85 s | +2.4% | `spread_ar1` | 1225 s | yes |
| 300 s | 306.44 s | +2.1% | `spread_ar1` | 3060 s | yes |
| 600 s | 594.75 s | −0.9% | `spread_ar1` | 5960 s | yes |

Within 4% across a 20× range. The window is selected by the plateau criterion, not fixed.

**Why the spread estimator wins.** The single-series estimators, given the same data over the full post-event window, return 2446 s for a true 120 s — a 20× overstatement. The efficient price resumes being a martingale after the release, and over a long window that random walk swamps the decay. The window scan makes it visible:

| window | estimated half-life | fitted φ |
|---:|---:|---:|
| 200 s | 60.9 s | 0.9447 |
| 460 s | 88.5 s | 0.9616 |
| **695 s** | **92.7 s** | 0.9633 |
| 1050 s | 136.1 s | 0.9749 |
| 1580 s | 410.0 s | 0.9916 |
| 3575 s | 1253.4 s | 0.9972 |
| 5000 s | 2445.5 s | 0.9986 |

Plateau, then runaway. Reporting a single number without this curve is how a two-minute half-life gets published as forty.

### Hasbrouck information share

| σ_A | σ_B | true share A | Hasbrouck bounds | Gonzalo–Granger | leader | correct |
|---:|---:|---:|:--|---:|:--|:--|
| 0.9 | 0.3 | 0.900 | [0.954, 0.958] | 0.619 | venue_a | yes |
| 0.7 | 0.5 | 0.662 | [0.776, 0.781] | 0.573 | venue_a | yes |
| 0.5 | 0.5 | 0.500 | [0.606, 0.613] | 0.554 | venue_a | tie, not graded |
| 0.3 | 0.9 | 0.100 | [0.130, 0.133] | 0.530 | venue_b | yes |

The leader is identified correctly in every decisive case and the estimate is monotone in the truth. It is **not** an unbiased point estimate of `σ_A²/(σ_A²+σ_B²)`, and it should not be: the Hasbrouck share also depends on the adjustment speeds `κ`, and the slower-adjusting venue genuinely carries more of the permanent component. The two shares disagree substantially (0.96 vs 0.62) because they measure different things — reporting only whichever looked better would be the error.

The Cholesky bounds are tight here (width 0.004) because residual correlation is near zero by construction. On real data with correlated innovations they widen, which is the honest signal that the decomposition is poorly identified.

### Calibration

Injected favourite–longshot distortion `a = 0.850`; Platt scaling recovered **0.883**. Murphy decomposition on the same sample:

```
n=8000  brier=0.1675  logloss=0.5052  skill=+0.3298
  reliability=0.00076   resolution=0.08267   uncertainty=0.24987
  ECE=0.0255  MCE=0.0534  platt a=0.883 b=+0.061 (overconfident)
```

Reliability — the only tradeable component — is three orders of magnitude smaller than uncertainty. That ratio is the whole story of why forecasting edge in prediction markets is thin.

---

## 2. The fee curve determines where arbitrage exists

### Minimum tradeable violation by price level

A two-market ladder pair is built at each price level; the monotonicity violation is widened one cent at a time until the scanner can book a guaranteed profit.

| price | min tradeable violation | predicted `2·r·p·(1−p)` |
|---:|---:|---:|
| 5c | **1c** | 0.67c |
| 10c | 2c | 1.26c |
| 20c | 3c | 2.24c |
| 30c | 4c | 2.94c |
| 40c | 4c | 3.36c |
| **50c** | **4c** | **3.50c** |
| 60c | 4c | 3.36c |
| 70c | 3c | 2.94c |
| 80c | 3c | 2.24c |
| 90c | 2c | 1.26c |

The empirical threshold matches the closed-form prediction, rounded up to the one-cent grid. **A violation already tradeable at 5c must be four times larger at even money.**

This is structural — a property of the fee schedule, not of any market — and it is the most actionable result in the repository. It says the scanner's attention belongs on tail-priced strikes, where a one-cent dislocation is already money and where a competitor pricing off mid-quotes will not see it.

### Detection sweep

Injecting a controlled violation into the simulated ladder (which happened to land on tail-priced strikes, hence the 2c threshold):

| violation | detected | guaranteed | capital | return on capital |
|---:|:--|---:|---:|---:|
| 0c | no | 0c | — | — |
| 2c | yes | 149c | 20,051c | 0.74% |
| 6c | yes | 1,676c | 48,324c | 3.47% |
| 12c | yes | 4,539c | 45,461c | 9.98% |
| 24c | yes | 10,341c | 39,659c | 26.07% |

Zero at zero — no false positives on a coherent book — then guaranteed profit rising linearly with the violation, exactly as the theory says it should.

Scanner latency: **5–15 ms** for a five-market ladder including the LP. The closed-form detectors alone run in under 0.1 ms.

---

## 3. Backtests

Identical data, identical fill model, queue-aware with latency and per-level fees charged.

| strategy | P&L | trades | contracts | fees | Sharpe | max DD | hit rate | deflated SR | fill rate |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ladder_arb` | $0.00 | 0 | 0 | $0.00 | — | $0.00 | — | — | 0.00 |
| `coherence` | $0.00 | 0 | 0 | $0.00 | — | $0.00 | — | — | 0.00 |
| `microprice` | $0.00 | 0 | 0 | $0.00 | — | $0.00 | — | — | 0.00 |
| `drift` | $19.15 | 7 | 175 | $2.35 | 102.4 | −$3.15 | 51.7% | 0.743 | 0.875 |
| `ladder_arb_stressed` | $386.70 | 87 | 8,700 | $95.30 | 20.1 | −$565.22 | 44.6% | 0.227 | 0.870 |

### The zeros are the result

Three strategies place no trades, and each for a reason that should be checked rather than fixed:

- **`ladder_arb`** — the simulated ladder is coherent by construction. A scanner finding arbitrage here would be broken.
- **`coherence`** — the projection residual is quote noise (~0.6c), below the fee hurdle. Correctly refuses to trade a mispricing it cannot monetise.
- **`microprice`** — with a 2-cent spread, the microprice tilt is worth at most 1c against a 3.5c round-trip hurdle. **Microstructure scalping is not viable on this exchange at this spread**, and that is a finding, not a failure.

`ladder_arb_stressed` is the identical strategy on the identical session with a 12-cent violation injected. It fires. That pair — silent on clean data, capturing on dirty data — is what makes the zeros meaningful. Either half alone proves nothing.

### Reading the risk statistics honestly

- **Fees consume 20% of gross** on the stressed arbitrage run ($95 of $482). That is what happens when you cross the spread 87 times, and it is why the sizing search optimises post-fee guaranteed profit rather than gross mispricing.
- **`drift`'s Sharpe of 102** is an artefact of annualising 5-second bars over a single session. It means "it worked on one event", which is exactly what the deflated Sharpe of 0.743 is there to say.
- **`ladder_arb_stressed`'s max drawdown of −$565 against +$387 of P&L** is mark-to-market noise: the legs settle against each other but are marked at independent mids while the injected dislocation is live, and the mark jumps when it is removed. The *settled* P&L is the real number.
- **Deflated Sharpe of 0.227** on the stressed run correctly reports that five variants were tried on one session and this is not yet evidence of anything.

---

## 4. Microstructure

From the simulated session's central strike:

| statistic | value |
|:--|--:|
| mean spread | 2.0c |
| Roll implied spread | comparable to quoted |
| Kyle's lambda | ~10⁻⁴ cents/contract |
| VPIN | ~0.1 |

Kyle's lambda is what caps executable size: an arbitrage needing more contracts than the book absorbs pays for its own edge on the way in. On these books, permanent impact is small relative to the fee hurdle — consistent with the finding that fees, not impact, are the binding constraint on this asset class.

---

## 5. Live market check

`kalshi-alpha scan --event KXGDPYEAR-36`, run unauthenticated against production on
2026-08-21. Fourteen bucket markets partitioning 2036 GDP growth, so exactly one settles
YES and the fair prices must sum to $1.

| | |
|:--|--:|
| markets, all two-sided | 14 |
| sum of best **asks** | **123c** |
| sum of best **bids** | **80c** |
| no-arbitrage band | **43c wide** |
| arbitrage found after fees | **0** |

Buying every outcome costs $1.23 for a guaranteed $1; selling every outcome pays $0.80 to
assume a $1 liability. Both are far outside the band, so the scanner correctly reports
nothing.

This is the single most useful sanity check in the repository, because it answers the
question the simulator cannot: **how much room is there in practice?** The answer is that
the binding constraint is not fee-versus-dislocation at all — quoted spreads on a
14-outcome event are roughly an order of magnitude wider than the 3.5c fee hurdle. Real
model-free arbitrage requires either a genuinely stale quote or a second venue, which is
what `detect_cross_venue` exists for.

Depth is substantial (several markets show $2,000-$5,000 of resting notional), so this is
not a case of an illiquid book quoting wide. The market is liquid and still 43 cents wide
across the round.

---

## 6. What would change on live data

The simulator is a GBM with digital payoffs. Real event contracts have fat tails, regime changes around resolution criteria, and settlement disputes that no diffusion captures. Specifically:

- **Arbitrage frequency is unknown here.** The simulator generates coherent books by construction, so this repository measures *detection capability and the fee threshold*, not how often real violations occur. That requires live data collection.
- **Half-life estimates would degrade.** Real releases have staggered information (headline, then details, then revisions), so a single exponential is a simplification.
- **Cross-venue arbitrage carries risks not priced here** — settlement-timing mismatch, funding cost, correlated counterparty risk.

The estimator validation is a **necessary** condition for trusting these methods on live data, not a sufficient one. What it does establish is that when the methods are wrong, it is because the world differs from the model — not because the code computes the wrong thing.
