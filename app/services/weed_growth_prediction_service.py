from app.schemas.weed_growth_prediction import PredictionResponse


class WeedGrowthPredictionService:
    def __init__(self):
        pass

    def generate_prediction(self) -> PredictionResponse:
        return PredictionResponse(message="Prediction succeeded")
