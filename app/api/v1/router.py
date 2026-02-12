from fastapi import APIRouter

from app.api.v1.endpoints import health, items, users

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(users.router)
api_router.include_router(items.router)
