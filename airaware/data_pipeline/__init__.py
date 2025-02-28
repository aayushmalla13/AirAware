"""Automated data pipeline for continuous data collection and updates."""

from .auto_updater import AutoDataUpdater, UpdateConfig

__all__ = [
    "AutoDataUpdater",
    "UpdateConfig",
]
