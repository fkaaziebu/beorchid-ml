from pydantic import BaseModel


class PredictionResponse(BaseModel):
    message: str


class PredictionInput(BaseModel):
    image_url: str
