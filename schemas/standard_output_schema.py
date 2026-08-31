from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class EmotionEvent(BaseModel):
    """Represents a single emotion detection event at a specific point in time."""

    timestamp_sec: float = Field(
        ...,
        description="Timestamp in seconds within the video when this emotion was detected.",
    )
    emotion: str = Field(
        ...,
        description="The predicted emotion label (e.g., 'Happy', 'Sad', 'Angry').",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score for the predicted emotion, ranging from 0.0 to 1.0.",
    )


class EmotionSummary(BaseModel):
    """Aggregated emotion percentages across the entire video."""

    Angry: float = Field(default=0.0, description="Percentage of frames classified as Angry.")
    Disgusted: float = Field(default=0.0, description="Percentage of frames classified as Disgusted.")
    Fearful: float = Field(default=0.0, description="Percentage of frames classified as Fearful.")
    Happy: float = Field(default=0.0, description="Percentage of frames classified as Happy.")
    Neutral: float = Field(default=0.0, description="Percentage of frames classified as Neutral.")
    Sad: float = Field(default=0.0, description="Percentage of frames classified as Sad.")
    Surprised: float = Field(default=0.0, description="Percentage of frames classified as Surprised.")


class AnalysisMetadata(BaseModel):
    """Metadata about the analysis run."""

    video_name: str = Field(
        ..., description="Name of the analyzed video file."
    )
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 timestamp of when the analysis was performed.",
    )
    total_frames_processed: int = Field(
        default=0,
        description="Total number of video frames that were processed.",
    )
    total_faces_detected: int = Field(
        default=0,
        description="Total number of face detections across all processed frames.",
    )
    video_duration_sec: Optional[float] = Field(
        default=None,
        description="Duration of the video in seconds, if available.",
    )


class StandardizedEmotionOutput(BaseModel):
    """
    Standardized output format for facial emotion analysis results.

    This schema is designed to provide a structured, time-aware, and
    metadata-rich output that can be consistently integrated into
    RUXAILAB reports and dashboards, following the goals of the
    'Sentiment and Emotion Output Standardization' GSoC project.
    """

    metadata: AnalysisMetadata = Field(
        ..., description="Metadata about the analysis run."
    )
    timeline: List[EmotionEvent] = Field(
        default_factory=list,
        description="Chronologically ordered list of per-frame emotion detection events.",
    )
    summary: EmotionSummary = Field(
        default_factory=EmotionSummary,
        description="Aggregated emotion percentages across the entire video.",
    )
