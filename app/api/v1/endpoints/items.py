from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.item import ItemCreate, ItemResponse, ItemUpdate
from app.services.item_service import ItemService

router = APIRouter(prefix="/items", tags=["items"])


@router.get("/", response_model=List[ItemResponse])
def list_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    service = ItemService(db)
    return service.list_items(skip=skip, limit=limit)


@router.post("/", response_model=ItemResponse, status_code=201)
def create_item(data: ItemCreate, owner_id: int, db: Session = Depends(get_db)):
    service = ItemService(db)
    return service.create_item(data, owner_id=owner_id)


@router.get("/{item_id}", response_model=ItemResponse)
def get_item(item_id: int, db: Session = Depends(get_db)):
    service = ItemService(db)
    return service.get_item(item_id)


@router.patch("/{item_id}", response_model=ItemResponse)
def update_item(item_id: int, data: ItemUpdate, db: Session = Depends(get_db)):
    service = ItemService(db)
    return service.update_item(item_id, data)


@router.delete("/{item_id}", status_code=204)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    service = ItemService(db)
    service.delete_item(item_id)
