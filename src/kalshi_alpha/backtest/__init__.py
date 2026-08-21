"""Event-driven backtesting with realistic fills."""

from kalshi_alpha.backtest.engine import Backtester, BacktestResult, Context
from kalshi_alpha.backtest.fills import FillModel, RestingOrder
from kalshi_alpha.backtest.metrics import (
    PerformanceStats,
    deflated_sharpe_ratio,
    max_drawdown,
    performance,
    sharpe_ratio,
)
from kalshi_alpha.backtest.strategies import (
    CoherenceStrategy,
    DriftStrategy,
    LadderArbStrategy,
    MicropriceStrategy,
    Strategy,
    build_strategy,
)

__all__ = [
    "BacktestResult",
    "Backtester",
    "CoherenceStrategy",
    "Context",
    "DriftStrategy",
    "FillModel",
    "LadderArbStrategy",
    "MicropriceStrategy",
    "PerformanceStats",
    "RestingOrder",
    "Strategy",
    "build_strategy",
    "deflated_sharpe_ratio",
    "max_drawdown",
    "performance",
    "sharpe_ratio",
]
