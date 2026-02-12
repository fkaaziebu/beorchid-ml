from app.schemas.disease_prediction import PredictionResponse


class DiseasePredictionService:
    def __init__(self):
        pass

    def generate_prediction(self) -> PredictionResponse:
        return PredictionResponse(message="Prediction succeeded")
