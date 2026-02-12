from fastapi import APIRouter

from app.api.v1.endpoints import (
    disease_prediction,
    health,
    items,
    users,
    weed_growth_prediction,
    yield_prediction,
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(users.router)
api_router.include_router(items.router)
api_router.include_router(disease_prediction.router)
api_router.include_router(weed_growth_prediction.router)
api_router.include_router(yield_prediction.router)
