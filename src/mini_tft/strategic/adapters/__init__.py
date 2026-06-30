"""Adapters over the strategic simulator core.

Adapters translate the shared strategic rules into Puffer, MuZero-cache,
playable-demo, and baseline surfaces. They must not fork simulator rules.
"""

from mini_tft.strategic.adapters import analytics, baselines, muzero_cache, puffer, web_demo

__all__ = ["analytics", "baselines", "muzero_cache", "puffer", "web_demo"]
