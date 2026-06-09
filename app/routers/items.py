from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import Pagination, pagination
from core.db import get_db
from app.exceptions import NotFoundError
from app.models import Item
from app.schemas import ItemCreate, ItemRead

router = APIRouter(prefix="/items", tags=["items"])


@router.get("", response_model=list[ItemRead])
async def list_items(
    page: Pagination = Depends(pagination),
    db: AsyncSession = Depends(get_db),
) -> list[Item]:
    result = await db.execute(
        select(Item).order_by(Item.id).limit(page.limit).offset(page.offset)
    )
    return list(result.scalars().all())


@router.post("", response_model=ItemRead, status_code=status.HTTP_201_CREATED)
async def create_item(
    payload: ItemCreate,
    db: AsyncSession = Depends(get_db),
) -> Item:
    item = Item(name=payload.name, description=payload.description)
    db.add(item)
    await db.flush()
    await db.refresh(item)
    return item


@router.get("/{item_id}", response_model=ItemRead)
async def get_item(
    item_id: int,
    db: AsyncSession = Depends(get_db),
) -> Item:
    item = await db.get(Item, item_id)
    if item is None:
        raise NotFoundError(f"item {item_id} 不存在")
    return item
