from fastapi import APIRouter

from app.schemas.weed_growth_prediction import PredictionResponse
from app.services.weed_growth_prediction_service import WeedGrowthPredictionService

router = APIRouter(prefix="/weed-growth", tags=["weed-growth"])


@router.post("/generate-prediction", response_model=PredictionResponse, status_code=200)
def generate_weed_growth_prediction():
    service = WeedGrowthPredictionService()
    return service.generate_prediction()
