from fastapi import APIRouter

from app.schemas.yield_prediction import PredictionResponse
from app.services.yield_prediction_service import YieldPredictionService

router = APIRouter(prefix="/yield", tags=["yield"])


@router.post("/generate-prediction", response_model=PredictionResponse, status_code=200)
def generate_yield_prediction():
    service = YieldPredictionService()
    return service.generate_prediction()
