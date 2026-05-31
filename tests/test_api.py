"""Tests for the Quantum API."""

import pytest
from httpx import AsyncClient, ASGITransport
from api.server import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_chat_not_trained():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/chat", json={"message": "Hello Quantum"})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_custom_params():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/chat", json={
            "message": "Hello",
            "max_new_tokens": 128,
            "temperature": 0.5,
        })
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_empty_message():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/chat", json={"message": ""})
    assert response.status_code in [200, 400]