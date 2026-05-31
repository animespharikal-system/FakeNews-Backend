from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["news article text"])


class PredictionResponse(BaseModel):
    prediction: str = Field(..., examples=["Fake"])
    confidence: float = Field(..., ge=0, le=100, examples=[95.2])
