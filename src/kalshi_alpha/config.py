"""Typed, environment-driven configuration.

Everything the engine needs is expressed here so that a run is reproducible from
a single object: fee schedule, risk limits, execution assumptions, and the
statistical settings used by the diffusion estimators.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

REST_BASE = {
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
}
WS_BASE = {
    "prod": "wss://api.elections.kalshi.com/trade-api/ws/v2",
    "demo": "wss://demo-api.kalshi.co/trade-api/ws/v2",
}


class FeeConfig(BaseModel):
    """Kalshi's fee schedule.

    The taker fee is a *concave* function of price::

        fee = ceil(rate * C * P * (1 - P))

    with ``P`` in dollars and the result rounded up to the next cent. It peaks
    at P = 0.50 (1.75c per contract at the 7% rate) and vanishes at the tails,
    which is why cheap-tail arbitrage legs are far more forgiving than legs
    struck near even money.
    """

    taker_rate: float = 0.07
    maker_rate: float = 0.0025
    maker_fees_enabled: bool = False
    settlement_fee_rate: float = 0.0

    @field_validator("taker_rate", "maker_rate", "settlement_fee_rate")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("fee rates must be non-negative")
        return v


class RiskConfig(BaseModel):
    """Pre-trade risk limits enforced by :mod:`kalshi_alpha.execution.risk`."""

    max_contracts_per_market: int = 5_000
    max_gross_exposure_cents: int = 2_000_000  # $20,000 notional
    max_open_orders: int = 200
    max_daily_loss_cents: int = 100_000  # $1,000
    max_order_qty: int = 1_000
    kill_switch_drawdown_cents: int = 150_000
    per_event_concentration_cents: int = 500_000


class ExecutionConfig(BaseModel):
    """Latency and fill assumptions shared by the backtester and paper broker."""

    order_latency_ms: float = 25.0
    market_data_latency_ms: float = 10.0
    cancel_latency_ms: float = 20.0
    # Fraction of the queue ahead of us that we assume is cancelled rather than
    # filled as trades print (queue "leakage"; 0 = pessimistic, 1 = optimistic).
    queue_leakage: float = 0.15
    # Probability an aggressive order is rejected because the book moved first.
    adverse_reject_prob: float = 0.02
    allow_partial_arb_legs: bool = False


class ArbConfig(BaseModel):
    """Thresholds for the arbitrage scanner."""

    min_edge_cents: int = 1  # guaranteed cents per unit portfolio
    min_qty: int = 1
    max_qty: int = 500
    max_legs: int = 12
    depth_levels: int = 5
    # Haircut applied to displayed size to reflect stale/phantom liquidity.
    size_haircut: float = 0.5
    require_two_sided: bool = True
    include_maker_legs: bool = False


class DiffusionConfig(BaseModel):
    """Settings for information-diffusion estimation."""

    pre_event_window_s: float = 900.0  # 15 min before
    post_event_window_s: float = 3600.0  # 60 min after
    bar_seconds: float = 5.0
    halflife_max_s: float = 1800.0
    vecm_lags: int = 5
    variance_ratio_qs: tuple[int, ...] = (2, 4, 8, 16, 32)
    bootstrap_draws: int = 500
    min_observations: int = 60


class Settings(BaseModel):
    """Top-level runtime configuration."""

    mode: str = "offline"  # "offline" | "live"
    env: str = "demo"  # "demo" | "prod"
    api_key_id: str | None = None
    private_key_path: Path | None = None
    data_dir: Path = Field(default=Path("data"))
    artifacts_dir: Path = Field(default=Path("artifacts"))
    log_level: str = "INFO"
    seed: int = 7

    fees: FeeConfig = Field(default_factory=FeeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    arb: ArbConfig = Field(default_factory=ArbConfig)
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in ("offline", "live"):
            raise ValueError("mode must be 'offline' or 'live'")
        return v

    @field_validator("env")
    @classmethod
    def _env(cls, v: str) -> str:
        if v not in ("demo", "prod"):
            raise ValueError("env must be 'demo' or 'prod'")
        return v

    @property
    def rest_base(self) -> str:
        return REST_BASE[self.env]

    @property
    def ws_base(self) -> str:
        return WS_BASE[self.env]

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key_id) and self.private_key_path is not None

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, **overrides: object) -> Settings:
        """Build settings from environment variables, then apply overrides."""
        pk = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        base: dict[str, object] = {
            "mode": os.getenv("KALSHI_ALPHA_MODE", "offline"),
            "env": os.getenv("KALSHI_ENV", "demo"),
            "api_key_id": os.getenv("KALSHI_API_KEY_ID") or None,
            "private_key_path": Path(pk) if pk else None,
            "data_dir": Path(os.getenv("KALSHI_ALPHA_DATA_DIR", "data")),
            "artifacts_dir": Path(os.getenv("KALSHI_ALPHA_ARTIFACTS_DIR", "artifacts")),
            "log_level": os.getenv("KALSHI_ALPHA_LOG_LEVEL", "INFO"),
        }
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)  # type: ignore[arg-type]


def load_settings(**overrides: object) -> Settings:
    return Settings.from_env(**overrides)
