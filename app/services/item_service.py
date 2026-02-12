from typing import List

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.item import Item
from app.repositories.item_repository import ItemRepository
from app.schemas.item import ItemCreate, ItemUpdate


class ItemService:
    def __init__(self, db: Session):
        self.repo = ItemRepository(db)

    def get_item(self, item_id: int) -> Item:
        item = self.repo.get_by_id(item_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Item not found",
            )
        return item

    def list_items(self, skip: int = 0, limit: int = 100) -> List[Item]:
        return self.repo.get_all(skip=skip, limit=limit)

    def create_item(self, data: ItemCreate, owner_id: int) -> Item:
        item = Item(
            title=data.title,
            description=data.description,
            price=data.price,
            owner_id=owner_id,
        )
        return self.repo.create(item)

    def update_item(self, item_id: int, data: ItemUpdate) -> Item:
        item = self.get_item(item_id)
        updates = data.model_dump(exclude_unset=True)
        return self.repo.update(item, updates)

    def delete_item(self, item_id: int) -> None:
        item = self.get_item(item_id)
        self.repo.delete(item)
