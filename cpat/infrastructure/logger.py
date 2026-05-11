"""Week 5 logging facade for live/system logging."""

from cpat.infrastructure.live_logger import LiveTradeLogger
from cpat.infrastructure.logging import setup_logging

__all__ = ["LiveTradeLogger", "setup_logging"]
