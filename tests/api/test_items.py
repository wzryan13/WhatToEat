import pytest


@pytest.mark.asyncio
async def test_list_items_empty(client):
    resp = await client.get("/api/v1/items")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_create_then_get_item(client):
    create = await client.post(
        "/api/v1/items", json={"name": "番茄炒蛋", "description": "家常菜"}
    )
    assert create.status_code == 201
    created = create.json()
    assert created["name"] == "番茄炒蛋"
    item_id = created["id"]

    got = await client.get(f"/api/v1/items/{item_id}")
    assert got.status_code == 200
    assert got.json()["id"] == item_id


@pytest.mark.asyncio
async def test_get_missing_item_returns_unified_error(client):
    resp = await client.get("/api/v1/items/99999")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "not_found"
    assert "message" in body
