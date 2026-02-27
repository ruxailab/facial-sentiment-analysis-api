from abc import ABC, abstractmethod

from schemas.emotion_schema import GetEmotionPercentagesResponse
from schemas.standard_output_schema import StandardizedEmotionOutput


class EmotionsAnalysisService(ABC):
    @abstractmethod
    def get_emotion_percentages(self, video_path: str) -> GetEmotionPercentagesResponse:
        pass

    @abstractmethod
    def get_standardized_output(self, video_path: str, video_name: str = "") -> StandardizedEmotionOutput:
        """
        Analyze a video and return a standardized output that includes
        metadata, a chronological timeline of emotion events with
        confidence scores, and an aggregated summary.
        """
        pass
