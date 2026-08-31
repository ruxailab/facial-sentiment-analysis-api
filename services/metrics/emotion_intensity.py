"""
Emotion intensity metrics for facial sentiment analysis.

This module computes derived intensity metrics from raw emotion
percentage data, enabling richer quantitative analysis of user
emotional responses during usability testing sessions.

It addresses the **"Confidence and Intensity Metrics"** key feature
of the GSoC 2026 project *"Sentiment and Emotion Output
Standardization for Usability Reports"*.

Metrics provided
----------------
- **Emotional Valence Score**: A single [-1, +1] value summarizing
  overall positivity vs. negativity of the emotional response.
- **Emotional Arousal Score**: A [0, 1] value indicating how
  emotionally activated (vs. calm/neutral) the user was.
- **Emotion Diversity Index**: Shannon entropy-based measure of how
  spread out the emotions are (0 = single emotion, 1 = uniform).
- **Dominant Emotion Strength**: How strongly the top emotion
  dominates over the runner-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ── Emotion valence mapping ──────────────────────────────────────────
# Each emotion is assigned a valence weight in [-1, +1].
# Positive emotions get positive weights, negative emotions get
# negative weights, and Neutral is zero.

VALENCE_WEIGHTS: dict[str, float] = {
    "Happy": 1.0,
    "Surprised": 0.3,      # mildly positive / ambiguous
    "Neutral": 0.0,
    "Sad": -0.7,
    "Fearful": -0.8,
    "Angry": -0.9,
    "Disgusted": -0.85,
}

# Arousal mapping: how "activated" each emotion is (0 = calm, 1 = high)
AROUSAL_WEIGHTS: dict[str, float] = {
    "Happy": 0.7,
    "Surprised": 0.9,
    "Neutral": 0.1,
    "Sad": 0.3,
    "Fearful": 0.8,
    "Angry": 0.9,
    "Disgusted": 0.6,
}


# ── Data class ───────────────────────────────────────────────────────

@dataclass
class EmotionIntensityMetrics:
    """Container for all computed intensity metrics."""

    valence_score: float = field(
        metadata={"description": "Overall emotional valence from -1 (negative) to +1 (positive)."}
    )
    arousal_score: float = field(
        metadata={"description": "Emotional activation level from 0 (calm) to 1 (high arousal)."}
    )
    diversity_index: float = field(
        metadata={"description": "Shannon entropy-based diversity from 0 (single emotion) to 1 (uniform)."}
    )
    dominant_emotion: str = field(
        metadata={"description": "The emotion label with the highest percentage."}
    )
    dominant_percentage: float = field(
        metadata={"description": "Percentage of the dominant emotion."}
    )
    dominance_ratio: float = field(
        metadata={"description": "Ratio of dominant emotion to the runner-up (>1 means clear dominance)."}
    )
    interpretation: str = field(
        metadata={"description": "Brief human-readable interpretation of the metrics."}
    )

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary for JSON responses."""
        return {
            "valence_score": round(self.valence_score, 4),
            "arousal_score": round(self.arousal_score, 4),
            "diversity_index": round(self.diversity_index, 4),
            "dominant_emotion": self.dominant_emotion,
            "dominant_percentage": round(self.dominant_percentage, 2),
            "dominance_ratio": round(self.dominance_ratio, 2),
            "interpretation": self.interpretation,
        }


# ── Core computation functions ───────────────────────────────────────

def _normalize_percentages(percentages: dict[str, float]) -> dict[str, float]:
    """Normalize percentages so they sum to 1.0 (proportions)."""
    total = sum(percentages.values())
    if total == 0:
        return {k: 0.0 for k in percentages}
    return {k: v / total for k, v in percentages.items()}


def compute_valence(proportions: dict[str, float]) -> float:
    """Compute weighted emotional valence score in [-1, +1].

    The valence is the weighted sum of each emotion's proportion
    multiplied by its valence weight.
    """
    score = sum(
        proportions.get(emotion, 0.0) * weight
        for emotion, weight in VALENCE_WEIGHTS.items()
    )
    return max(-1.0, min(1.0, score))


def compute_arousal(proportions: dict[str, float]) -> float:
    """Compute weighted emotional arousal score in [0, 1].

    The arousal is the weighted sum of each emotion's proportion
    multiplied by its arousal weight.
    """
    score = sum(
        proportions.get(emotion, 0.0) * weight
        for emotion, weight in AROUSAL_WEIGHTS.items()
    )
    return max(0.0, min(1.0, score))


def compute_diversity(proportions: dict[str, float]) -> float:
    """Compute Shannon entropy-based emotion diversity index in [0, 1].

    A value of 0 means all weight is on a single emotion.
    A value of 1 means perfectly uniform distribution.
    The raw Shannon entropy is normalized by log(n) where n is the
    number of non-zero emotions.
    """
    non_zero = {k: v for k, v in proportions.items() if v > 0}
    n = len(non_zero)
    if n <= 1:
        return 0.0

    entropy = -sum(p * math.log(p) for p in non_zero.values())
    max_entropy = math.log(n)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_dominance_ratio(percentages: dict[str, float]) -> tuple[str, float, float]:
    """Return (dominant_emotion, dominant_pct, dominance_ratio).

    The dominance ratio is the percentage of the top emotion divided
    by the percentage of the runner-up. A high ratio indicates a
    clearly dominant emotion.
    """
    sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
    if not sorted_items:
        return ("None", 0.0, 0.0)

    dominant = sorted_items[0]
    runner_up_pct = sorted_items[1][1] if len(sorted_items) > 1 else 0.0

    ratio = dominant[1] / runner_up_pct if runner_up_pct > 0 else float("inf")
    return (dominant[0], dominant[1], ratio)


def _interpret(valence: float, arousal: float, diversity: float,
               dominant: str, dominance_ratio: float) -> str:
    """Generate a brief interpretation of the intensity metrics."""
    parts: list[str] = []

    # Valence interpretation
    if valence > 0.3:
        parts.append("The emotional response is predominantly positive.")
    elif valence < -0.3:
        parts.append("The emotional response is predominantly negative.")
    else:
        parts.append("The emotional response is relatively balanced between positive and negative.")

    # Arousal interpretation
    if arousal > 0.6:
        parts.append("Users showed high emotional activation, indicating strong engagement or reaction.")
    elif arousal < 0.3:
        parts.append("Users showed low emotional activation, suggesting a calm or disengaged state.")
    else:
        parts.append("Emotional activation was moderate.")

    # Diversity interpretation
    if diversity > 0.8:
        parts.append("Emotions were highly diverse, with no single emotion dominating.")
    elif diversity < 0.4:
        parts.append(f"The emotional profile was concentrated, with {dominant} being clearly dominant.")
    else:
        parts.append("There was moderate emotional diversity across the session.")

    return " ".join(parts)


# ── Public API ───────────────────────────────────────────────────────

def compute_intensity_metrics(
    percentages: dict[str, float],
) -> EmotionIntensityMetrics:
    """Compute all emotion intensity metrics from raw percentages.

    Parameters
    ----------
    percentages:
        Dictionary mapping emotion labels (e.g. ``"Happy"``) to their
        percentage values (0–100).

    Returns
    -------
    EmotionIntensityMetrics
        Dataclass containing all computed metrics.
    """
    proportions = _normalize_percentages(percentages)
    valence = compute_valence(proportions)
    arousal = compute_arousal(proportions)
    diversity = compute_diversity(proportions)
    dominant, dominant_pct, dom_ratio = compute_dominance_ratio(percentages)
    interpretation = _interpret(valence, arousal, diversity, dominant, dom_ratio)

    return EmotionIntensityMetrics(
        valence_score=valence,
        arousal_score=arousal,
        diversity_index=diversity,
        dominant_emotion=dominant,
        dominant_percentage=dominant_pct,
        dominance_ratio=dom_ratio,
        interpretation=interpretation,
    )
