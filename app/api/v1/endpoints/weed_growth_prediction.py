from fastapi import APIRouter

from app.schemas.prediction import PredictionInput, PredictionResponse
from app.services.weed_growth_prediction_service import WeedGrowthPredictionService

router = APIRouter(prefix="/weed-growth", tags=["weed-growth"])


@router.post("/generate-prediction", response_model=PredictionResponse, status_code=200)
def generate_weed_growth_prediction(data: PredictionInput):
    service = WeedGrowthPredictionService()
    return service.generate_prediction(data)
