"""Strategy discovery and UI-facing metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass

from cpat.config.loader import CPATConfig


@dataclass(frozen=True)
class StrategyDescriptor:
    """Minimal strategy metadata needed by the UI."""

    key: str
    label: str
    enabled: bool
    config_path: str
    supports_live: bool = False


class StrategyService:
    """Expose discoverable strategy metadata without executing strategies."""

    def list_strategies(self, config: CPATConfig) -> list[StrategyDescriptor]:
        """Return configured strategies and their current enabled flags."""
        return [
            StrategyDescriptor(
                key="momentum",
                label="Cross-Sectional Momentum",
                enabled=config.strategies.momentum.enabled,
                config_path="strategies.momentum",
                supports_live=True,
            ),
            StrategyDescriptor(
                key="mean_reversion",
                label="Mean Reversion",
                enabled=config.strategies.mean_reversion.enabled,
                config_path="strategies.mean_reversion",
                supports_live=False,
            ),
            StrategyDescriptor(
                key="both",
                label="Momentum + Mean Reversion",
                enabled=True,
                config_path="strategies",
                supports_live=False,
            ),
        ]

    def strategy_parameter_paths(self, strategy: str) -> tuple[str, ...]:
        """Return the dotted-path parameter namespace for a strategy."""
        mapping = {
            "momentum": ("strategies.momentum",),
            "mean_reversion": ("strategies.mean_reversion",),
            "both": ("strategies.momentum", "strategies.mean_reversion"),
        }
        return mapping.get(strategy, (f"strategies.{strategy}",))
