from __future__ import annotations

import mini_tft.strategic as strategic
from mini_tft.strategic import adapters, core
from mini_tft.strategic.adapters import analytics, baselines, muzero_cache, puffer, web_demo


def test_strategic_namespace_scaffold_imports() -> None:
    assert strategic.__all__ == ["adapters", "core"]
    assert adapters.__all__ == ["analytics", "baselines", "muzero_cache", "puffer", "web_demo"]
    assert "reset" in core.__all__
    assert "step" in core.__all__
    assert "StrategicAction" in core.__all__
    assert "summarize_episode_rows" in analytics.__all__
    assert "tft_heuristic_policy" in baselines.__all__
    assert "generate_cache" in muzero_cache.__all__
    assert "run_benchmark" in puffer.__all__
    assert web_demo.__all__ == ["state_payload"]
