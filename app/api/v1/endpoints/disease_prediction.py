from fastapi import APIRouter

from app.schemas.prediction import PredictionInput, PredictionResponse
from app.services.disease_prediction_service import DiseasePredictionService

router = APIRouter(prefix="/disease", tags=["disease"])


@router.post("/generate-prediction", response_model=PredictionResponse, status_code=200)
async def generate_disease_prediction(data: PredictionInput):
    service = DiseasePredictionService()
    return await service.generate_prediction(data)
