"""
Standardized output schema for multimodal sentiment fusion.

Converts the facial-sentiment-analysis-api emotion percentage output
into the dict format expected by MultimodalSentimentEngine.analyze()
in the sentiment-analysis-api repo.

Usage:
    from schemas.multimodal_output_schema import FacialEmotionOutput

    result = GetEmotionPercentagesResponse(...)
    payload = FacialEmotionOutput.from_response(result)
    # payload.to_multimodal_dict() → ready for /multimodal/analyze
"""

from pydantic import BaseModel, Field
from schemas.emotion_schema import GetEmotionPercentagesResponse


class FacialEmotionOutput(BaseModel):
    """
    Wraps GetEmotionPercentagesResponse and exposes a conversion method
    that produces the facial_emotions dict expected by the
    MultimodalSentimentEngine.

    The MultimodalSentimentEngine expects:
        {emotion_label: percentage_float, ...}
    where percentage values are in the range 0.0 to 100.0 and labels
    match the 7-class taxonomy used by the facial CNN model.
    """

    Angry:     float = Field(ge=0.0, le=100.0)
    Disgusted: float = Field(ge=0.0, le=100.0)
    Fearful:   float = Field(ge=0.0, le=100.0)
    Happy:     float = Field(ge=0.0, le=100.0)
    Neutral:   float = Field(ge=0.0, le=100.0)
    Sad:       float = Field(ge=0.0, le=100.0)
    Surprised: float = Field(ge=0.0, le=100.0)

    @classmethod
    def from_response(
        cls, response: GetEmotionPercentagesResponse
    ) -> "FacialEmotionOutput":
        """
        Build a FacialEmotionOutput from a GetEmotionPercentagesResponse.

        Args:
            response: the Pydantic model returned by EmotionsAnalysisImp

        Returns:
            FacialEmotionOutput instance ready for to_multimodal_dict()
        """
        return cls(
            Angry=response.Angry,
            Disgusted=response.Disgusted,
            Fearful=response.Fearful,
            Happy=response.Happy,
            Neutral=response.Neutral,
            Sad=response.Sad,
            Surprised=response.Surprised,
        )

    def to_multimodal_dict(self) -> dict:
        """
        Convert to the facial_emotions dict format expected by
        MultimodalSentimentEngine.analyze().

        Returns:
            dict of {emotion_label: percentage_float}
            Example:
                {
                    'Angry': 2.5,
                    'Disgusted': 0.0,
                    'Fearful': 1.2,
                    'Happy': 61.3,
                    'Neutral': 28.0,
                    'Sad': 5.0,
                    'Surprised': 2.0
                }
        """
        return {
            'Angry':     self.Angry,
            'Disgusted': self.Disgusted,
            'Fearful':   self.Fearful,
            'Happy':     self.Happy,
            'Neutral':   self.Neutral,
            'Sad':       self.Sad,
            'Surprised': self.Surprised,
        }

    def dominant_emotion(self) -> tuple:
        """
        Return the emotion with the highest percentage and its value.

        Returns:
            (label: str, percentage: float)
            Example: ('Happy', 61.3)
        """
        emotions = self.to_multimodal_dict()
        label = max(emotions, key=emotions.get)
        return label, emotions[label]
