"""API endpoint tests."""
import pytest


@pytest.mark.asyncio
async def test_health_liveness(api_client):
    resp = await api_client.get("/health/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_readiness(api_client):
    resp = await api_client.get("/health/ready")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_deep(api_client):
    resp = await api_client.get("/health/deep")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "components" in data


@pytest.mark.asyncio
async def test_list_streams_empty(api_client):
    resp = await api_client.get("/streams/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_add_stream_requires_auth(api_client):
    resp = await api_client.post("/streams/", json={
        "stream_id": "test-cam",
        "source": "rtsp://test",
    })
    # Should require auth
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_get_events_empty(api_client):
    resp = await api_client.get("/events/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_get_events_stats(api_client):
    resp = await api_client.get("/events/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "by_type" in data


@pytest.mark.asyncio
async def test_query_without_vlm_returns_503(api_client):
    # Get a token first
    login_resp = await api_client.post("/auth/token",
                                       json={"username": "admin", "password": "vortex-admin-pass"})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]

    resp = await api_client.post(
        "/query/",
        json={"question": "Show me all red cars"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_auth_token_invalid_credentials(api_client):
    resp = await api_client.post("/auth/token",
                                 json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_rate_limiting(api_client):
    """Verify rate limiter doesn't block health checks."""
    for _ in range(10):
        resp = await api_client.get("/health/")
        assert resp.status_code == 200
