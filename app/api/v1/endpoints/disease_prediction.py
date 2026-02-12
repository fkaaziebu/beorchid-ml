from fastapi import APIRouter

from app.schemas.disease_prediction import PredictionResponse
from app.services.disease_prediction_service import DiseasePredictionService

router = APIRouter(prefix="/disease", tags=["disease"])


@router.post("/generate-prediction", response_model=PredictionResponse, status_code=200)
def generate_disease_prediction():
    service = DiseasePredictionService()
    return service.generate_prediction()
