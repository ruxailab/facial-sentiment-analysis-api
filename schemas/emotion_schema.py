from pydantic import BaseModel
from typing import List


class GetEmotionPercentagesResponse(BaseModel):
    Angry: float
    Disgusted: float
    Fearful: float
    Happy: float
    Neutral: float
    Sad: float
    Surprised: float


class TimelineEntry(BaseModel):
    id: int
    starting_time: str
    ending_time: str
    emotion: str
    value: float


class EmotionAnalysisResponse(BaseModel):
    emotions: GetEmotionPercentagesResponse
    timeline: List[TimelineEntry]