
import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_post_response():
    """
    Fixture to provide a mock response object for POST requests.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"answer": "This is a mocked answer."}
    return mock_resp

@pytest.fixture
def valid_questions():
    """
    Fixture to provide a list of valid questions for the /ask endpoint.
    """
    return [f"What is the answer to question {i}?" for i in range(50)]

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_high_load_ask_endpoint(mock_post_response, valid_questions):
    """
    Performance test: Simulates 50 concurrent POST requests to /ask endpoint.
    Verifies all responses are successful, no 5xx errors, and 95th percentile response time < 5000ms.
    All HTTP calls are mocked to avoid real network connections.
    """
    # Patch requests.post to always return a mocked successful response
    with patch("requests.post", return_value=mock_post_response) as mock_post:
        import requests  # Import inside patch context to ensure patching

        endpoint_url = "http://localhost:8080/ask"

        response_times = []

        async def send_request(question):
            start = time.time()
            # Simulate sending a POST request to /ask
            resp = requests.post(endpoint_url, json={"question": question})
            elapsed = (time.time() - start) * 1000  # ms
            response_times.append(elapsed)
            return resp

        # Run 50 concurrent requests
        tasks = [send_request(q) for q in valid_questions]
        responses = await asyncio.gather(*tasks)

        # Assert all requests completed successfully
        assert len(responses) == 50, "Not all requests completed"

        # Assert no 5xx errors
        for resp in responses:
            assert 200 <= resp.status_code < 300, f"Unexpected status code: {resp.status_code}"

        # Calculate 95th percentile response time
        sorted_times = sorted(response_times)
        idx_95 = int(0.95 * len(sorted_times)) - 1
        percentile_95 = sorted_times[idx_95]

        assert percentile_95 < 5000, (
            f"95th percentile response time is {percentile_95:.2f}ms, expected < 5000ms"
        )

        # Ensure the mock was called 50 times
        assert mock_post.call_count == 50, f"requests.post called {mock_post.call_count} times, expected 50"

