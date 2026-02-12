from fastapi import APIRouter

from app.schemas.prediction import PredictionInput, PredictionResponse
from app.services.yield_prediction_service import YieldPredictionService

router = APIRouter(prefix="/yield", tags=["yield"])


@router.post("/generate-prediction", response_model=PredictionResponse, status_code=200)
def generate_yield_prediction(data: PredictionInput):
    service = YieldPredictionService()
    return service.generate_prediction(data)
