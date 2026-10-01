"""Live event-based focus meter (lens focus audit instrument).

The metric, pane workers, rendering, and session logging behind the
ops repo's focus meter — the tool used to set and lock the focus ring
of every ring/rig camera against a modulated far target.
"""

from evio.focus.metric import (
    FocusResult,
    ScoreHistory,
    WindowScorer,
    bin_count_image,
    hf_ratio,
    score_window,
)

__all__ = [
    "FocusResult",
    "ScoreHistory",
    "WindowScorer",
    "bin_count_image",
    "hf_ratio",
    "score_window",
]
