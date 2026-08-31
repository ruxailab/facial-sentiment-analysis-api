"""
Human-readable summary generator for facial emotion analysis results.

This module converts raw emotion percentage data into natural-language
summaries that non-technical stakeholders (UX researchers, designers,
product managers) can immediately understand and act upon.

It addresses the **"Human-Readable Report Summaries"** key feature
of the GSoC 2026 project *"Sentiment and Emotion Output
Standardization for Usability Reports"*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ── Emotion category helpers ─────────────────────────────────────────

POSITIVE_EMOTIONS = {"Happy", "Surprised"}
NEGATIVE_EMOTIONS = {"Angry", "Disgusted", "Fearful", "Sad"}
NEUTRAL_EMOTIONS = {"Neutral"}

EMOTION_DESCRIPTORS: dict[str, str] = {
    "Happy": "happiness",
    "Surprised": "surprise",
    "Angry": "anger",
    "Disgusted": "disgust",
    "Fearful": "fear",
    "Sad": "sadness",
    "Neutral": "neutral affect",
}


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class EmotionBreakdown:
    """Convenience wrapper around a dict of emotion → percentage."""

    percentages: dict[str, float]

    # -- derived helpers --------------------------------------------------

    @property
    def dominant_emotion(self) -> str:
        """Return the emotion label with the highest percentage."""
        return max(self.percentages, key=self.percentages.get)  # type: ignore[arg-type]

    @property
    def dominant_percentage(self) -> float:
        return self.percentages[self.dominant_emotion]

    @property
    def positive_total(self) -> float:
        return sum(v for k, v in self.percentages.items() if k in POSITIVE_EMOTIONS)

    @property
    def negative_total(self) -> float:
        return sum(v for k, v in self.percentages.items() if k in NEGATIVE_EMOTIONS)

    @property
    def neutral_total(self) -> float:
        return sum(v for k, v in self.percentages.items() if k in NEUTRAL_EMOTIONS)

    def top_n(self, n: int = 3) -> list[tuple[str, float]]:
        """Return the *n* emotions with the highest percentages."""
        sorted_items = sorted(self.percentages.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]


# ── Summary generation ───────────────────────────────────────────────

def _sentiment_label(breakdown: EmotionBreakdown) -> str:
    """Return an overall sentiment label: Positive / Negative / Mixed / Neutral."""
    pos = breakdown.positive_total
    neg = breakdown.negative_total
    neu = breakdown.neutral_total

    if neu >= 60:
        return "Neutral"
    if pos >= 50 and neg < 15:
        return "Positive"
    if neg >= 50 and pos < 15:
        return "Negative"
    if pos > neg and pos >= 30:
        return "Mostly Positive"
    if neg > pos and neg >= 30:
        return "Mostly Negative"
    return "Mixed"


def _top_emotions_sentence(breakdown: EmotionBreakdown) -> str:
    """Build a natural-language sentence listing the top emotions."""
    top = breakdown.top_n(3)
    # Filter out emotions with 0%
    top = [(e, p) for e, p in top if p > 0]
    if not top:
        return "No emotions were detected during the analysis."

    parts: list[str] = []
    for emotion, pct in top:
        descriptor = EMOTION_DESCRIPTORS.get(emotion, emotion.lower())
        parts.append(f"{descriptor} ({pct:.1f}%)")

    if len(parts) == 1:
        return f"The dominant emotion was {parts[0]}."
    elif len(parts) == 2:
        return f"The top emotions were {parts[0]} and {parts[1]}."
    else:
        return f"The top emotions were {parts[0]}, {parts[1]}, and {parts[2]}."


def _ux_insight(breakdown: EmotionBreakdown) -> str:
    """Provide a brief UX-oriented insight based on the emotion profile."""
    sentiment = _sentiment_label(breakdown)
    dominant = breakdown.dominant_emotion

    if sentiment in ("Positive", "Mostly Positive"):
        return (
            "The overall emotional response was positive, suggesting that "
            "users found the experience engaging or satisfying."
        )
    elif sentiment in ("Negative", "Mostly Negative"):
        if dominant == "Angry":
            return (
                "The prevalence of anger suggests frustration, possibly "
                "caused by usability issues or unmet expectations."
            )
        elif dominant == "Fearful":
            return (
                "Elevated fear levels may indicate user anxiety or confusion "
                "during the interaction."
            )
        elif dominant == "Sad":
            return (
                "Sadness was the dominant emotion, which may reflect "
                "disappointment or a sense of loss during the experience."
            )
        elif dominant == "Disgusted":
            return (
                "Disgust was prominent, potentially pointing to content or "
                "design elements that users found off-putting."
            )
        return (
            "The overall emotional response was negative, indicating "
            "potential usability or experience issues that warrant further investigation."
        )
    elif sentiment == "Neutral":
        return (
            "The emotional response was predominantly neutral, which may "
            "indicate a straightforward, low-engagement interaction."
        )
    else:
        return (
            "The emotional response was mixed, with both positive and "
            "negative signals. A deeper qualitative review is recommended."
        )


def generate_summary(
    percentages: dict[str, float],
    video_name: Optional[str] = None,
) -> dict[str, str]:
    """Generate a human-readable summary from emotion percentages.

    Parameters
    ----------
    percentages:
        A dictionary mapping emotion labels (e.g. ``"Happy"``) to their
        percentage values (0–100).
    video_name:
        Optional video file name to include in the summary header.

    Returns
    -------
    dict with the following keys:

    - ``"overall_sentiment"`` – one-word / short-phrase sentiment label.
    - ``"top_emotions"`` – natural-language sentence about the top emotions.
    - ``"ux_insight"`` – a brief UX-oriented interpretation.
    - ``"full_summary"`` – a combined paragraph suitable for reports.
    """
    breakdown = EmotionBreakdown(percentages=percentages)
    sentiment = _sentiment_label(breakdown)
    top_sentence = _top_emotions_sentence(breakdown)
    insight = _ux_insight(breakdown)

    header = ""
    if video_name:
        header = f"Analysis of \"{video_name}\": "

    full_summary = (
        f"{header}The overall emotional sentiment is **{sentiment}**. "
        f"{top_sentence} {insight}"
    )

    return {
        "overall_sentiment": sentiment,
        "top_emotions": top_sentence,
        "ux_insight": insight,
        "full_summary": full_summary,
    }
