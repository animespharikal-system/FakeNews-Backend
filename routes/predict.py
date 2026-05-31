from fastapi import APIRouter

from models.prediction import PredictionRequest, PredictionResponse
from services.prediction_service import PredictionService


router = APIRouter(tags=["Prediction"])
prediction_service = PredictionService()


@router.post("/predict", response_model=PredictionResponse)
def predict_news(request: PredictionRequest) -> PredictionResponse:
    return prediction_service.predict(request.text)
