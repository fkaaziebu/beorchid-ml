from pydantic import BaseModel


class PredictionResponse(BaseModel):
    message: str
