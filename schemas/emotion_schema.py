from pydantic import BaseModel

class GetEmotionPercentagesResponse(BaseModel):
    Angry: float
    Disgusted: float
    Fearful: float
    Happy: float
    Neutral: float
    Sad: float
    Surprised: float