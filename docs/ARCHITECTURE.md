# Architecture

## Dependency direction

Strictly one-way. Nothing below imports anything above it.

```
                          types.py  ·  config.py  ·  logging_setup.py
                                          │
        ┌──────────────┬──────────────────┼──────────────────┬──────────────┐
        │              │                  │                  │              │
  microstructure   probability        arbitrage            data          report
   book/features    calibration     fees → liquidity     client/store     html
   impact           aggregation     → payoff → detectors  synthetic
                    constraints     → lp → engine         events
        │              │                  │                  │
        └──────────────┴────────┬─────────┴──────────────────┘
                                │
                          backtest  ·  execution
                                │
                               cli
```

`types.py` has no internal dependencies at all — it is the vocabulary everything else speaks. `config.py` depends only on pydantic. Circular imports are impossible by construction, and the one place a late import appears (`payoff` inside `detectors._best_size`) is a deliberate hot-loop optimisation, not a cycle.

## Module responsibilities

| module | owns | does **not** own |
|---|---|---|
| `types` | book mechanics, settlement cash flows, position netting | anything requiring numpy |
| `config` | every tunable, typed and validated | any I/O |
| `arbitrage/fees` | the exchange fee schedule, in one place | book state |
| `arbitrage/liquidity` | walking a book with exact per-level fees | deciding what to walk |
| `arbitrage/payoff` | proving a portfolio profitable state by state | finding portfolios |
| `arbitrage/detectors` | fast named-structure scans | proving them (delegates to `payoff`) |
| `arbitrage/lp` | the general programme and its dual | naming structures |
| `arbitrage/engine` | orchestration, dedupe, latency accounting | any detection logic |
| `microstructure/book` | L2 reconstruction, sequence integrity | interpreting the book |
| `diffusion/*` | estimators, each independently testable | data acquisition |
| `probability/*` | scoring, recalibration, coherence projection | trading decisions |
| `backtest/fills` | what a fill actually costs | when to trade |
| `backtest/engine` | the time loop and its ordering guarantees | strategy logic |
| `execution/risk` | saying no | placing orders |
| `execution/oms` | knowing what is working at the exchange | deciding what to send |
| `data/synthetic` | ground truth | any estimator |

The split between `detectors` and `payoff` is the important one. Detectors *propose*; `payoff` *disposes*. A detector cannot emit an opportunity that has not been proved profitable in every settlement state, because the only constructor is `build_opportunity`, which returns `None` unless the worst state clears the bar.

## The time loop

The backtester's ordering is the property that makes it a backtest rather than a demonstration.

At each timestamp, in this order:

1. **Arrivals.** Orders whose `submit_ts + latency` has passed reach the engine and trade against *this* bar's book. Marketable limits cross; GTC remainders rest; IOC/FOK remainders are cancelled.
2. **Passive fills.** Resting orders are matched against this bar's tape, consuming queue-ahead first, then swept through if the touch has moved past them.
3. **Strategy.** Called with the book as of `t − market_data_latency`. It cannot see step 1 or 2 of a future bar.
4. **Mark.** Equity is cash plus marked positions plus the unconditional NO-settlement credit.

At the end, open positions **settle** against the realised outcome rather than being marked out.

There is no code path by which a decision consumes information from its own future.

## Configuration

One `Settings` object carries the entire run: fee schedule, risk limits, execution assumptions, arbitrage thresholds, estimator settings, seed. A run is reproducible from that object alone.

Defaults are deliberately conservative:

- `mode = "offline"` — importing the package can never reach the network.
- `size_haircut = 0.5` — displayed size is assumed half real.
- `require_two_sided = True` — one-sided books are excluded from structural scans.
- `maker_fees_enabled = False` — matching Kalshi's default, but explicit rather than assumed.

Environment variables override defaults; explicit keyword arguments override the environment.

## Error handling

Three different postures, chosen per situation:

**Fail loudly** where continuing corrupts state. A websocket sequence gap raises `SequenceGap` and forces a re-subscribe, because quoting against a book that has diverged from the exchange is how phantom arbitrages get "found" and traded. An illegal OMS state transition raises rather than being coerced.

**Fail quietly and record** where a partial result is still useful. A failed orderbook fetch logs and drops that ticker rather than aborting a hundred-market scan.

**Return "unknown" rather than a number** where the data cannot support an answer. `select_window` returns `reliable=False`; `implied_state_prices` returns `None` when infeasible; `parse_market_meta` returns `strike_type=None` for a market whose settlement rule it cannot parse, and such markets are excluded from structural scans rather than guessed at.

## Concurrency

The REST client is async with a shared token bucket, so a hundred concurrent book fetches self-limit rather than being throttled. Retries are jittered exponential backoff, and only on idempotent verbs — a retried POST could double a position, so order placement carries a client-generated idempotency key and is retried only when the server explicitly signals nothing was created.

Everything else is synchronous. The analysis code is numpy-bound and the scanner runs in single-digit milliseconds; adding threads would buy nothing and cost determinism.

## Storage

Book snapshots are stored **flattened** — one row per `(ts, ticker)` with the top N levels as columns. Nested lists are more faithful to the wire format but make every downstream query pay deserialisation, and the analysis overwhelmingly wants columns.

Partitioning is by event ticker and UTC date, matching the actual access pattern ("every book for CPI on release day"), so a study touches a handful of files rather than scanning the corpus.

## Testing strategy

Four layers, each catching what the others cannot:

1. **Exact-value tests** for the fee schedule. Not tolerance tests — an off-by-one-cent error propagates into every arbitrage decision.
2. **Property tests** (hypothesis) stating the no-arbitrage theorems directly. "Every reported arbitrage is profitable in every settlement state" is a claim about all inputs, so it needs a test that searches for a counterexample rather than one that checks three cases.
3. **Ground-truth recovery** against the simulator. The only layer that can catch a *correct-looking but wrong* estimator.
4. **False-positive tests.** A scanner that finds edge which is not there loses money on every signal it generates, so "silent on a coherent book" is tested harder than "fires on a violation".

Three real bugs were caught by layers 3 and 4 during development, all documented in [METHODOLOGY](METHODOLOGY.md): the position-netting identity, event-study re-basing, and marketable-limit detection. Each produced plausible-looking output while being wrong.

## Extending

**A new arbitrage structure** needs a settlement matrix (`{ticker: [0/1 per state]}`) and a leg builder. Everything else — sizing search, fee accounting, proof, dedupe — is shared. If the structure is expressible as a payoff matrix, the LP already finds it and a named detector is only a latency optimisation.

**A new venue** needs a client exposing `snapshot() -> dict[str, OrderBook]`. `detect_cross_venue` and the price-discovery estimators take it from there.

**A new strategy** is any callable `Context -> Sequence[Order]`. Register it in `build_strategy` to reach the CLI.
