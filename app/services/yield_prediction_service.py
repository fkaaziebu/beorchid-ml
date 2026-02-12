from app.schemas.yield_prediction import PredictionResponse


class YieldPredictionService:
    def __init__(self):
        pass

    def generate_prediction(self) -> PredictionResponse:
        return PredictionResponse(message="Prediction succeeded")
