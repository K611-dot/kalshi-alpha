# Methodology

The maths behind each component, and — more importantly — why each modelling choice was made rather than the obvious alternative.

---

## 1. The instrument

A Kalshi contract pays $1 if an event occurs and $0 otherwise. Prices live on a 1..99 cent integer grid. The exchange publishes **two bid books** per market, YES and NO, and there is no separate ask book because

```
buy  YES @ p  ≡  sell NO  @ (100 − p)
sell YES @ p  ≡  buy  NO  @ (100 − p)
```

so the YES offer ladder is the mirror of the NO bid ladder. We model that faithfully rather than flattening to a synthetic bid/ask, because the mirror relation has a direct consequence.

### Why "buy YES + buy NO < $1" cannot happen inside one market

The YES ask is *defined* as `100 − (best NO bid)`. So

```
ask_YES + ask_NO = ask_YES + (100 − bid_YES) = 100 + spread ≥ 100
```

The naive complement arbitrage is structurally impossible on a single market. Any hit from `detect_crossed_book` therefore means the book is crossed or stale — valuable as a data-integrity alarm, occasionally real for a few milliseconds after a large sweep, but never a standing opportunity. The genuine complement edge lives *across venues*, which is why `detect_cross_venue` exists as a separate detector. This is asserted as a property test over randomly generated books.

### Prices in integer cents

All P&L accounting is in integer cents. Probabilities are floats only at the modelling boundary. This is not fastidiousness: an arbitrage is a claim about a portfolio being profitable in *every* state, and a claim that survives only because of floating-point slack is not a claim at all.

---

## 2. Fees

```
F = ceil( r · C · P · (1 − P) )        r = 0.07 on most markets
```

with `P` in dollars and the result rounded up to the next cent, **per order**.

Two properties drive everything downstream.

**It is concave, peaking at even money.** 1.75c per contract at 50c; 0.33c at 5c. A four-leg structure assembled from tail-priced legs survives on an edge that would be entirely consumed if the same legs were struck near 50c.

**The ceiling is per order, not per contract.** Rounding up is a fixed cost amortised over size, so small clips are disproportionately penalised. `exact_taker_fee_cents` models this exactly with integer arithmetic (bit-reproducible across platforms); `linear_taker_fee_cents` drops the ceiling so the arbitrage LP stays linear. The linear version is a strict *under*-estimate by at most one cent per order, so it is optimistic — which is why every LP solution is re-priced with the exact function before being reported.

**The round-trip hurdle.** Entering and exiting as a taker near the same price costs `2·r·P·(1−P)` cents per contract:

| price | hurdle |
|---:|---:|
| 5c | 0.67c |
| 10c | 1.26c |
| 30c | 2.94c |
| **50c** | **3.50c** |
| 90c | 1.26c |

Any signal not compared against this number is not a signal. It is also why the `microprice` strategy correctly places zero trades against a 2-cent spread: the microprice tilt can be worth at most one cent, and one cent does not clear 3.5.

---

## 3. Arbitrage

### 3.1 Canonical leg form

Every leg the detectors emit is a **buy**. Shorting YES is expressed as buying NO at the mirrored price, which is how the matching engine executes it anyway, and it removes all ambiguity about collateral and settlement cash flow.

### 3.2 The structures

**Dutch book over a mutually exclusive group.**

- *Under-round* (`Σ ask < $1`): buy one YES of every outcome; exactly one pays $1. This requires the group to be **exhaustive**. If some outcome is missing, the portfolio can pay nothing and it is a punt, not an arbitrage — a distinction the code enforces via `EventGroup.exhaustive`.
- *Over-round* (`Σ bid > $1`): buy one NO of every outcome; at least `n−1` pay $1. This needs only mutual exclusivity, so it is available on partial groups too.

**Ladder monotonicity.** For a `≥ k` ladder the survival function must be non-increasing. For `k_i < k_j`, the pair `BUY YES(k_i) + BUY NO(k_j)` pays at least $1 in every state and $2 when the underlying lands between the strikes, so it is an arbitrage whenever the pair costs less than $1 all-in.

Two subtleties:

1. **All pairs are scanned, not just adjacent ones.** With bid/ask spreads the executable condition `ask_i < bid_j` is *not transitive*: `ask₁=40, bid₂=39, ask₂=42, bid₃=41` has no adjacent violation but `ask₁ = 40 < 41 = bid₃` is one. Adjacent-only scanning misses these.
2. **Digital ladders require only first-order monotonicity.** Unlike vanilla option strikes, there is no butterfly/convexity constraint to enforce, because the second difference of a survival function is an unconstrained density increment.

### 3.3 The general LP

Drop the names. Each **price level** of each side of each market is a separate instrument, so the LP walks the book by itself — two levels with identical payoff differ only in cost, and the solver exhausts the cheaper one first. That converts a convex piecewise-linear execution-cost curve into plain linear algebra.

With `xᵢ ≥ 0` contracts, all-in cost `cᵢ`, settlement payoff `A_ij`, displayed size `uᵢ`, budget `B`:

```
max  z
s.t. Σᵢ (A_ij − cᵢ) xᵢ  ≥  z      ∀ states j
     0 ≤ xᵢ ≤ uᵢ
     Σᵢ cᵢ xᵢ ≤ B
```

`z* > 0` is a certificate of arbitrage: a portfolio making at least `z*` cents whatever happens. Shorting needs no variable of its own, because buying the complementary NO contract *is* the short.

The continuous solution is then floored to whole contracts, re-priced with the **exact** ceiling fee, and re-verified state by state. Scaling down preserves hedge ratios, so a descending ladder of scales is searched and the first that still clears the bar is kept. If none does, the result is reported as `lost_to_rounding` rather than as an opportunity.

### 3.4 The dual: state prices and no-arbitrage bounds

By the fundamental theorem of asset pricing, no arbitrage exists **iff** there is a probability vector `q` over states with `A q ≤ c` for every tradeable instrument. `implied_state_prices` recovers one.

This is more useful than it first appears. Reading each market's mid independently gives numbers that usually do not describe any distribution — an exhaustive group summing to 1.04, a ladder that ticks up between strikes. Probabilities derived from the joint state prices are mutually consistent **by construction**.

Maximising and minimising `q · payoff` over the consistent set gives model-free **super- and sub-replication bounds** for an arbitrary claim. A wide band means the quoted surface simply does not pin the claim down; a narrow one means the market has already priced it. This is the honest answer to "what is this illiquid contract worth?"

### 3.5 Proof, not assertion

`build_opportunity` enumerates the settlement states and only returns an opportunity if the **worst** one is strictly profitable. That is the line between a mispricing (positive EV under a model) and an arbitrage (positive value under every state, with no model at all).

The empirical threshold for a ladder violation matches the closed-form prediction `2·r·p·(1−p)`, rounded up to the cent grid — 1c at 5c, 4c at even money. Arbitrage on this exchange is roughly four times cheaper to capture in the tails.

---

## 4. Information diffusion

### 4.1 Price discovery

Two venues quoting the same event are cointegrated: the spread is bounded by arbitrage. That makes "who moves first?" a well-posed statistical question. With the cointegrating vector known to be `(1, −1)`:

```
Δp_t = α (p₁,ₜ₋₁ − p₂,ₜ₋₁) + Σₖ Γₖ Δp_{t−k} + ε_t
```

Both equations share regressors, so SUR collapses to equation-by-equation OLS and the joint estimate is exact.

The adjustment vector `α` carries the answer: a venue that does **not** adjust (`αᵢ ≈ 0`) is setting the price. The orthogonal vector

```
ψ = (α₂, −α₁) / (α₂ − α₁)
```

gives the permanent shock's loading on each venue.

- **Gonzalo–Granger component share** `ψᵢ` — contribution to the permanent component. Depends only on error correction, not volatility.
- **Hasbrouck information share** — share of the efficient price's *innovation variance*, which also rewards being noisy-but-early:

```
ISⱼ = ([ψ M]ⱼ)² / (ψ Ω ψ')      Ω = M M'  (Cholesky)
```

**The bounds are the result, not a caveat.** `IS` is identified only up to the Cholesky ordering when innovations are correlated. With two series there are exactly two orderings, so the bounds are exact rather than approximated. Reporting a point estimate while hiding them is the most common error in this literature — and the bounds are widest precisely when contemporaneous correlation is high, i.e. when both venues are fast and the question is most interesting.

The two shares deliberately disagree (0.96 vs 0.62 on the same data in the demo): they measure different things, and presenting only whichever is more flattering would be the error.

### 4.2 Half-life of adjustment

Three estimators, chosen because they fail differently:

1. **Adjustment profile** — non-parametric. Rescale so pre-event is 0 and settled post-event is 1, read off `t50`/`t90`. Assumes no functional form; needs a trustworthy terminal price.
2. **Weighted exponential decay** — regress `log|p_t − p_∞|` on time, weighted by the gap itself. Without that weighting the tail — where the gap is pure noise around zero — dominates the log-space fit and biases `τ` upward, making every market look slower than it is.
3. **AR(1) on the gap** — no intercept, since the gap is defined relative to the terminal price and its unconditional mean is zero by construction.

#### The failure mode that matters

Measuring adjustment against a **fixed terminal price** assumes the efficient price stops moving. It does not: after the release it resumes being a martingale, and over a long window that random walk swamps the decay. The estimated half-life then grows roughly linearly in the window, without bound.

`window_scan` makes this visible — a plateau once the window covers several half-lives, then runaway. `select_window` finds the knee: among candidate windows spanning at least six half-lives of their own estimate (so a too-short window cannot win by being trivially self-consistent), pick the one whose estimate changes least when widened. **When no candidate qualifies, the result is returned as `reliable=False`** rather than dressed up as an answer.

#### The estimator that does not have this problem

When a leading series is available — another venue, or the underlying mapped into probability space — the gap between them is stationary by construction, because arbitrage bounds it. Its AR(1) coefficient measures transmission speed cleanly and is insensitive to window length. `information_lag_half_life` is therefore the preferred estimator, and when present it wins the consensus outright rather than being averaged with the single-series estimates, which would only import their bias.

Measured recovery: within 4% across a 20× range of true half-lives (30 s to 600 s).

### 4.3 Lead–lag and the Epps effect

The two series are **asynchronous**: the underlying ticks many times a second, the contract may sit unchanged for a minute. Sampling both onto a common grid induces the **Epps effect** — measured correlation collapses toward zero as the interval shrinks, purely as an artefact of non-synchronous observation, and the bias grows exactly as you zoom into the horizon you care about.

**Hayashi–Yoshida** avoids the grid entirely, summing products of returns over *overlapping* intervals only. It is unbiased for integrated covariance with no synchronisation step. Shifting one clock and re-estimating traces out a lead–lag curve whose peak is the propagation delay. The two-pointer sweep is O(n + m).

### 4.4 Event study

Scheduled releases are the cleanest natural experiment in finance: the timing is known in advance to the second, which removes the hardest problem in ordinary event studies — guessing when the information arrived.

1. **Align** onto an event-relative grid with *previous-tick* interpolation. Forward-filling would leak information backwards across the event boundary and manufacture the instant repricing being measured.
2. **Abnormalise** — re-base to the **last pre-event price**, then remove the pre-event drift rate. Basing at `t = 0` instead subtracts out precisely the jump the study exists to measure and reports a large real reaction as approximately zero. (This was a live bug, caught by a test that injected a known 12-cent jump and got 0.06 back.)
3. **Aggregate** into CAAR with cross-sectional standard errors.
4. **Falsify** with placebo timestamps drawn away from real events. With a handful of events and fat tails, the parametric t-stat is not credible; the placebo distribution assumes nothing. The exclusion radius adapts to the sample length — a fixed radius wide enough for a month of data silently yields *zero* draws on one session, and a p-value from zero draws is a missing result, not a conservative one.

### 4.5 Efficiency tests

**Variance ratio (Lo–MacKinlay).** `VR(q) > 1` means underreaction and continued drift; `< 1` means mean reversion or bid-ask bounce. The **heteroskedasticity-robust** statistic is the default because volatility on event contracts is wildly non-constant — it collapses as the price approaches 0 or 100 — so the homoskedastic version rejects essentially everywhere and tells you nothing.

The *shape* across `q` is the signal: drift that decays shows VR rising then falling as `q` passes the half-life, localising the absorption horizon with no functional-form assumption.

**Runs test** catches sign predictability when magnitudes are too heavy-tailed for variance-based tests. **Ljung–Box** is the portmanteau check that no single lag was cherry-picked.

---

## 5. Probability

### 5.1 Scoring

A market quote *is* a probability forecast. Murphy's decomposition separates what is tradeable from what is not:

```
BS = REL − RES + UNC
```

Only **reliability** (miscalibration) is tradeable. **Resolution** is skill you cannot capture; **uncertainty** is the variance of the world. A perfectly calibrated but useless forecaster quotes the base rate every time and has zero resolution.

Reliability curves use **quantile** bins by default — equal-width bins leave the tails nearly empty, and the tails are where the favourite–longshot bias lives — with **Wilson** intervals, which stay inside `[0,1]` near the boundaries where the normal approximation produces impossible bounds.

**Isotonic (PAVA)** gives a non-parametric monotone correction; **Platt scaling** gives two interpretable parameters — `a < 1` means the market is overconfident and should be shrunk toward 50c. Keeping both means you learn *what* is wrong, not just how to fix it.

### 5.2 Pooling

Averaging probabilities linearly is provably underconfident: the mean of two independent calibrated forecasts is less extreme than either. Pooling in **log-odds** fixes the direction; **extremization** (`odds → odds^a`, `a > 1`) corrects the residual shrinkage from shared information. A market quote and a model output are exactly two correlated forecasts of the same binary event, so this applies directly.

The default combiner gives the market most of the weight. That is a deliberate prior: on a liquid contract the quote already aggregates everyone's information, and a model overriding it has to earn the right to.

### 5.3 No-arbitrage projection

Projecting quoted probabilities onto the simplex (exhaustive groups) or the monotone cone (ladders) gives the closest *coherent* surface. Both projections are Euclidean, so nothing is imposed beyond coherence.

The **residual** is a continuous mispricing signal, available even when the dislocation is too small for fees to permit an actual arbitrage — which is most of the time. It is the same object the detectors hunt, seen from the modelling side: a negative bucket probability in `ladder_to_bucket_pmf` is a direct readout of the arbitrage that `detect_ladder_violation` would fire on.

---

## 6. Backtesting

### 6.1 What the fill model charges for

Most backtests assume a fill at the mid, instantly. That single assumption manufactures more phantom alpha than any modelling error, because it hands the strategy the spread it would have paid.

- **Latency** — an order submitted at `t` reaches the engine at `t + latency` and trades against the book *then*.
- **Queue position** — a passive order joins the back of the queue at its price and only fills after the resting size ahead is consumed. The most important detail for any quoting strategy, and the most often skipped.
- **Queue leakage** — some of the queue ahead cancels rather than trades, advancing us faster than raw volume implies.
- **Adverse selection** — an order the market trades *through* is filled at the worst possible moment. Modelling this is what stops a quoting strategy looking risk-free.
- **Marketable-limit detection** — a limit priced at or through the touch *crosses*; it does not rest. Treating every priced order as passive hands arbitrage legs a maker fill they never earned, and the resulting zero-fee front-of-queue execution flatters the strategy enormously. (Also a live bug, caught by a test asserting that arb fills are `taker`.)
- **Explicit settlement** — open positions are paid off against the realised outcome, not marked at a mid the strategy could not have traded out at.

### 6.2 The netting identity

Exposure is tracked in YES-equivalent contracts, because a NO contract is economically a short YES. That netting is correct for *directional* risk but drops a constant worth real money:

```
terminal = 100·s·Y + 100·(1−s)·N = 100·s·(Y − N) + 100·N
```

The `100·N` term is a **guaranteed** payment a long NO position collects however the event resolves. Holding one YES and one NO nets to zero exposure and still pays exactly $1 — which is precisely why crossed-book and ladder arbitrages are riskless. Netting exposure without carrying that term understates P&L by $1 per offsetting pair and makes every hedged structure look like a total loss. (Third live bug; a hedged arbitrage was reporting −$84 instead of +$16.)

### 6.3 Statistics that expose overfitting

A backtest Sharpe is not an estimate of future performance — it is the **maximum** of however many Sharpes were computed along the way, and the maximum of many noisy draws is biased upward substantially. Search twenty settings on pure noise and the best one reliably looks good.

- **Deflated Sharpe** (Bailey–López de Prado) returns the probability the true Sharpe is positive given the number of configurations actually tried, correcting also for skew and kurtosis. `n_trials` must be the honest count of every variant evaluated; understating it defeats the deflation.
- **Stationary bootstrap** (Politis–Romano) with geometric block lengths, preserving serial dependence. An i.i.d. bootstrap gives a spuriously narrow band.
- **Walk-forward splits are anchored**, never k-fold. K-fold cross-validation on time series leaks, badly, and is the second most common way a backtest lies.

---

## 7. Known limitations

- The simulator is a GBM with digital payoffs. Real event contracts have fat tails, regime changes, and settlement disputes that no diffusion captures. The estimator validation is therefore a **necessary** condition for correctness, not a sufficient one.
- `dt`-bucketed snapshot replay lets two orders in the same bar consume the same displayed size. The strategies mitigate this by merging legs per market, but a true matching engine would be better.
- Cross-venue detection models the second venue as a top-of-book quote plus a flat per-contract fee. Real cross-venue arbitrage also carries settlement-timing risk, funding cost, and correlated counterparty risk that this does not price.
- Half-life estimates degrade when the true half-life approaches the horizon over which residual volatility is stable. The code flags this (`identified=False`) rather than hiding it, but flagging is not solving.
- Kalshi's fee schedule and API surface change. The fee formula is centralised in `arbitrage/fees.py` and the endpoints in `config.py` for exactly that reason, but they need checking against current documentation before any live use.
