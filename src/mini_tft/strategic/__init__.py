"""Clean strategic MiniTFT proof lane.

This namespace is intentionally separate from the historical toy simulator in
``mini_tft.core``. New one-day proof work should enter through
``mini_tft.strategic.core`` and expose adapters under
``mini_tft.strategic.adapters``.
"""

from mini_tft.strategic import adapters, core

__all__ = ["adapters", "core"]
