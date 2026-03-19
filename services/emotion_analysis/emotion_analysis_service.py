from abc import ABC, abstractmethod

from schemas.emotion_schema import EmotionAnalysisResponse


class EmotionsAnalysisService(ABC):
    @abstractmethod
    def get_emotion_percentages(self, video_path: str, interval_s: int = 10) -> EmotionAnalysisResponse:
        pass