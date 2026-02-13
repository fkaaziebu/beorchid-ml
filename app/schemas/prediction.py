from typing import List

from pydantic import BaseModel


class Top3Prediction(BaseModel):
    label: str
    confidence: float


class LeafDetectionResult(BaseModel):
    bbox: List[float]
    detection_confidence: float
    predicted_disease: str
    confidence: float
    top3_predictions: List[Top3Prediction]


class PredictionResponse(BaseModel):
    message: str
    results: List[LeafDetectionResult] = []


class PredictionInput(BaseModel):
    image_url: str
