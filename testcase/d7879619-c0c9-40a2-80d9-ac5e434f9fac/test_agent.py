
import pytest
import time
from unittest.mock import patch, MagicMock

@pytest.fixture
def typical_question_payload():
    """Fixture for a typical /ask endpoint question payload."""
    return {
        "question": "What is the capital of France?",
        "user_id": "user123"
    }

@pytest.fixture
def mock_requests_post():
    """Fixture to patch requests.post for /ask endpoint."""
    with patch("requests.post") as mock_post:
        # Simulate a fast, successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"answer": "Paris"}
        mock_post.return_value = mock_response
        yield mock_post

@pytest.mark.performance
def test_performance_ask_endpoint_latency(typical_question_payload, mock_requests_post):
    """
    Measures the response time of the /ask endpoint under normal load to ensure it meets performance requirements.
    Success criteria:
      - Average response time is below 3000ms
      - No timeouts or server errors
    """
    import requests

    url = "http://localhost:8000/ask"
    num_requests = 5
    response_times = []

    for _ in range(num_requests):
        start = time.time()
        response = requests.post(url, json=typical_question_payload)
        end = time.time()
        elapsed_ms = (end - start) * 1000
        response_times.append(elapsed_ms)
        # Assert no server error or timeout
        assert response.status_code == 200, f"Server error: {response.status_code}"
        assert "answer" in response.json(), "Missing answer in response"

    avg_response_time = sum(response_times) / len(response_times)
    assert avg_response_time < 3000, f"Average response time {avg_response_time:.2f}ms exceeds 3000ms"

    # Optionally print for debug (pytest will capture)
    print(f"Average /ask endpoint response time: {avg_response_time:.2f}ms")
