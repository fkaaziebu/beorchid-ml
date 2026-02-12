from typing import List

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate, UserUpdate


class UserService:
    def __init__(self, db: Session):
        self.repo = UserRepository(db)

    def get_user(self, user_id: int) -> User:
        user = self.repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        return user

    def list_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        return self.repo.get_all(skip=skip, limit=limit)

    def create_user(self, data: UserCreate) -> User:
        existing = self.repo.get_by_email(data.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
        user = User(
            email=data.email,
            full_name=data.full_name,
            hashed_password=data.password,  # TODO: hash in production
        )
        return self.repo.create(user)

    def update_user(self, user_id: int, data: UserUpdate) -> User:
        user = self.get_user(user_id)
        updates = data.model_dump(exclude_unset=True)
        return self.repo.update(user, updates)

    def delete_user(self, user_id: int) -> None:
        user = self.get_user(user_id)
        self.repo.delete(user)
