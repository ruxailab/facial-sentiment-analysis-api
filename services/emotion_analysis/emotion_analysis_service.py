from abc import ABC, abstractmethod

from schemas.emotion_schema import GetEmotionPercentagesResponse

class EmotionsAnalysisService(ABC):
    @abstractmethod
    def get_emotion_percentages(self, video_path: str) -> GetEmotionPercentagesResponse:
        pass