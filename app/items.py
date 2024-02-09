from typing import List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter()


class Item(BaseModel):
    id: int
    name: str


storage: dict[int, Item] = {}


@router.get("/item/", response_model=List[Item])
async def read_items() -> List[Item]:
    return list(storage.values())


@router.post("/item/", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item) -> Item:
    if item.id in storage:
        raise HTTPException(status_code=400, detail="Item already exists")
    storage[item.id] = item
    return item


@router.get("/item/{item_id}/")
async def read_item(item_id: int) -> Item:
    item = storage.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.put("/item/{item_id}/")
async def update_item(item_id: int, item: Item) -> Item:
    if item_id not in storage:
        raise HTTPException(status_code=404, detail="Item not found")
    storage[item_id] = item
    return item


@router.delete("/item/{item_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int) -> None:
    if item_id not in storage:
        raise HTTPException(status_code=404, detail="Item not found")
    del storage[item_id]
    return
