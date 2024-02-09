import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from pytest_mock import MockFixture

from .. import items

app = FastAPI()
app.include_router(items.router)


client = TestClient(app)


@pytest.fixture
def mock_storage(mocker: MockFixture) -> list[items.Item]:
    test_items = [
        items.Item(id=1, name="test1"),
        items.Item(id=2, name="test2"),
    ]
    storage = {item.id: item for item in test_items}
    mocker.patch.object(items, "storage", storage)
    return test_items


def test_read_items(mock_storage: list[items.Item]) -> None:
    response = client.get("/item/")
    assert response.status_code == status.HTTP_200_OK
    expected_items = [item.model_dump() for item in mock_storage]
    assert response.json() == expected_items


def test_create_item() -> None:
    item_data = {"id": 3, "name": "test3"}
    response = client.post("/item/", json=item_data)
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json() == item_data


def test_create_item_if_exists(mock_storage: list[items.Item]) -> None:
    item_data = {"id": 1, "name": "test1"}
    response = client.post("/item/", json=item_data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json() == {"detail": "Item already exists"}


def test_read_item(mock_storage: list[items.Item]) -> None:
    item = mock_storage[0]
    response = client.get(f"/item/{item.id}/")
    assert response.status_code == 200
    assert response.json() == item.model_dump()


def test_read_item_if_does_not_exist() -> None:
    response = client.get("/item/1/")
    assert response.status_code == 404
    assert response.json() == {"detail": "Item not found"}


def test_update_item(mock_storage: list[items.Item]) -> None:
    item = mock_storage[0]
    new_item_data = {"id": 1, "name": "new test1"}
    response = client.put(f"/item/{item.id}/", json=new_item_data)
    assert response.status_code == 200
    assert response.json() == new_item_data


def test_update_item_if_does_not_exist() -> None:
    item_data = {"id": 1, "name": "test1"}
    response = client.put("/item/1/", json=item_data)
    assert response.status_code == 404
    assert response.json() == {"detail": "Item not found"}


def test_delete_item(mock_storage: list[items.Item]) -> None:
    item = mock_storage[0]
    id = item.id
    response = client.delete(f"/item/{id}/")
    assert response.status_code == 204
    assert id not in items.storage


def test_delete_item_if_does_not_exist():
    response = client.delete("/item/1/")
    assert response.status_code == 404
    assert response.json() == {"detail": "Item not found"}
