"""CPAT strategies package."""

from cpat.strategies.base import AbstractStrategy
from cpat.strategies.mean_reversion import MeanReversionStrategy
from cpat.strategies.momentum import MomentumStrategy

__all__ = ["AbstractStrategy", "MomentumStrategy", "MeanReversionStrategy"]
