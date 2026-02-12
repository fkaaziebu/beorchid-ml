from app.schemas.prediction import PredictionInput, PredictionResponse


class DiseasePredictionService:
    def __init__(self):
        pass

    def generate_prediction(self, data: PredictionInput) -> PredictionResponse:
        return PredictionResponse(message=f"Prediction succeeded #{data}")
