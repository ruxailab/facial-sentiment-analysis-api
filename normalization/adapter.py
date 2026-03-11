"""
Adapter that converts facial emotion analysis results into the
standardized RUXAILAB output schema.

This module bridges the gap between the emotion percentages produced
by this API and the unified schema used across the RUXAILAB platform
(text, audio, and facial analysis services).
"""

from datetime import datetime, timezone
from typing import Optional, Union

from schemas.emotion_schema import GetEmotionPercentagesResponse
from normalization.schema import StandardizedOutput, ResultEntry


# Mapping from the title-case labels used by this API to the lowercase
# labels defined in the standardized schema.
EMOTION_LABEL_MAP = {
    "Angry": "angry",
    "Disgusted": "disgusted",
    "Fearful": "fearful",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprised": "surprised",
}


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_emotions(
    emotion_data: Union[GetEmotionPercentagesResponse, dict],
    video_path: str,
    source_model: str = "facial-emotion-model2",
    task_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> StandardizedOutput:
    """Convert facial emotion percentages to the standardized schema.

    Parameters
    ----------
    emotion_data : GetEmotionPercentagesResponse or dict
        Emotion percentages as returned by the emotion analysis service.
        Values are expected in the 0-100 range (percentages) but 0-1
        values are also accepted.
    video_path : str
        Path or identifier of the video that was analyzed.
    source_model : str
        Identifier for the model that produced the result.
    task_id : str or None
        Optional task or session identifier for usability testing.
    timestamp : str or None
        ISO 8601 timestamp. If not provided the current UTC time is used.

    Returns
    -------
    StandardizedOutput
        A validated instance of the standardized schema.
    """
    if isinstance(emotion_data, GetEmotionPercentagesResponse):
        percentages = emotion_data.model_dump()
    else:
        percentages = dict(emotion_data)

    results = []
    for raw_label, score in percentages.items():
        normalized_label = EMOTION_LABEL_MAP.get(raw_label, raw_label.lower())
        # Convert 0-100 percentages to 0-1 scores
        if score > 1.0:
            score = score / 100.0
        results.append(
            ResultEntry(
                label=normalized_label,
                score=round(score, 4),
            )
        )

    # Sort by score descending so the dominant emotion comes first
    results.sort(key=lambda r: r.score, reverse=True)

    return StandardizedOutput(
        analysis_type="emotion",
        modality="facial",
        source_model=source_model,
        timestamp=timestamp or _now_iso(),
        task_id=task_id,
        input_summary=video_path,
        results=results,
    )
