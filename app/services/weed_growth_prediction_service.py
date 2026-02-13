from app.schemas.prediction import PredictionInput, PredictionResponse


class WeedGrowthPredictionService:
    def __init__(self):
        pass

    def generate_prediction(self, data: PredictionInput) -> PredictionResponse:
        # load model
        # get_image_raw_data
        # make_prediction
        # return prediction result
        return PredictionResponse(message=f"Prediction succeeded #{data}")
