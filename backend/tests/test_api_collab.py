"""API tests for /api/collab endpoints."""

from httpx import ASGITransport, AsyncClient

from neural_mri.main import app


async def test_create_session():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/collab/create?display_name=TestHost")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "host_id" in data
    assert "join_url" in data


async def test_get_session():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create_resp = await client.post("/api/collab/create?display_name=TestHost")
        session_id = create_resp.json()["session_id"]
        resp = await client.get(f"/api/collab/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id


async def test_get_nonexistent_session():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/collab/nonexistent")
    assert resp.status_code == 404


async def test_delete_session():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create_resp = await client.post("/api/collab/create?display_name=TestHost")
        session_id = create_resp.json()["session_id"]
        resp = await client.delete(f"/api/collab/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


async def test_list_sessions():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/collab/list")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
