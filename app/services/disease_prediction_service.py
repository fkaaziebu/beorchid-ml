from fastapi import HTTPException, status

from app.schemas.prediction import (
    LeafDetectionResult,
    PredictionInput,
    PredictionResponse,
    Top3Prediction,
)
from app.services.disease_prediction_pipeline import DiseasePredictionPipeline
from app.utils.image_utils import fetch_image


class DiseasePredictionService:
    def __init__(self):
        pass

    async def generate_prediction(self, data: PredictionInput) -> PredictionResponse:
        try:
            pipeline = DiseasePredictionPipeline()
            # load model
            pipeline.load_crop_classifier()
            # get_image_raw_data
            image_bytes, image = await fetch_image(data.image_url)
            # make_prediction
            ## detect leafs in image
            leaf_detections = pipeline.detect_leaves_in_image(image)
            # classify leaf disease for every leaf detection
            leaf_detection_result = []
            for i, detection in enumerate(leaf_detections):
                bbox = detection["bbox"]
                leaf_confidence = detection["confidence"]

                # preprocess leaf for classification
                leaf_image = pipeline.preprocess_leaf_for_classification(image, bbox)

                if leaf_image is None:
                    continue

                disease_class, confidence, top3 = pipeline.predict_leaf_disease(
                    leaf_image
                )

                # top3 may be an error string if classification failed
                top3_list = []
                if isinstance(top3, list):
                    top3_list = [
                        Top3Prediction(label=p["class"], confidence=p["confidence"])
                        for p in top3
                    ]

                result = LeafDetectionResult(
                    bbox=bbox,
                    detection_confidence=float(leaf_confidence),
                    predicted_disease=(disease_class or "UNKNOWN")
                    .upper()
                    .replace(" ", "_"),
                    confidence=float(confidence),
                    top3_predictions=top3_list,
                )
                leaf_detection_result.append(result)

            return PredictionResponse(
                message="Prediction succeeded",
                results=leaf_detection_result,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
