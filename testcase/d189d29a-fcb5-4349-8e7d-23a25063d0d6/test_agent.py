
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    """
    Fixture that returns a test client for the web application.
    This should be replaced with the actual client fixture from your app,
    e.g., FlaskClient, FastAPI TestClient, etc.
    """
    # Example for FastAPI:
    # from myapp import app
    # from fastapi.testclient import TestClient
    # return TestClient(app)
    #
    # For demonstration, we'll mock the client.
    mock_client = MagicMock()
    return mock_client

def mock_health_response(*args, **kwargs):
    """
    Returns a mock response object for the /health endpoint.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "status": "ok"
    }
    return mock_resp

def test_functional_health_check_endpoint(client):
    """
    Functional test: Ensures the /health endpoint returns a successful status and correct payload.
    - HTTP status code is 200
    - response['success'] is True
    - response['status'] == 'ok'
    """
    # Patch the client's get method to return a mock response
    with patch.object(client, "get", side_effect=mock_health_response) as mock_get:
        response = client.get("/health")
        assert response.status_code == 200, "Expected HTTP 200 for /health endpoint"
        data = response.json()
        assert data["success"] is True, "Expected 'success' to be True in response"
        assert data["status"] == "ok", "Expected 'status' to be 'ok' in response"
        mock_get.assert_called_once_with("/health")
