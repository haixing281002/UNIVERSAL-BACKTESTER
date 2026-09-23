"""Universal Backtester -- a small, dependency-light, causally-correct
long-only backtesting engine for cross-sectional and single/multi-asset
strategies.

Not tied to any fund, market, or data source. Point it at any wide
DatetimeIndex-by-asset price frame and a scoring/alpha frame and it will run
the strategy without letting a signal trade the bar that produced it.

    from universal_backtester import Backtester, build_allocator
    from universal_backtester.metrics import compute_metrics

See examples/momentum_rotation.py for a full worked strategy.
"""
from universal_backtester.engine import Backtester, BacktestResult, LookaheadError, assert_causal
from universal_backtester.allocators import Allocator, AllocatorContext, build_allocator, list_allocators

__all__ = [
    "Backtester", "BacktestResult", "LookaheadError", "assert_causal",
    "Allocator", "AllocatorContext", "build_allocator", "list_allocators",
]

__version__ = "0.1.0"
