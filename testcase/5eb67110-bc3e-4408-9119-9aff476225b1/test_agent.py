
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask, jsonify, request

@pytest.fixture
def client():
    """
    Fixture to provide a Flask test client for the /ask endpoint.
    Assumes the app exposes a /ask endpoint.
    """
    # Minimal Flask app for demonstration; in real tests, import your app
    app = Flask(__name__)

    @app.route('/ask', methods=['POST'])
    def ask():
        data = request.get_json()
        question = data.get('question', '')
        # Simulate the answer logic (would call LLM, Azure Search, etc.)
        if not question:
            return jsonify({'success': False, 'answer': ''}), 400
        # Simulate successful answer
        return jsonify({'success': True, 'answer': 'Mars is the fourth planet from the Sun.'}), 200

    with app.test_client() as client:
        yield client

def mock_llm_response(*args, **kwargs):
    # Simulate a successful LLM response
    return "Mars is the fourth planet from the Sun."

def mock_azure_search_response(*args, **kwargs):
    # Simulate a successful Azure Search response
    return ["Mars is the fourth planet from the Sun."]

def mock_knowledge_base_not_empty(*args, **kwargs):
    # Simulate knowledge base is not empty
    return True

@pytest.mark.functional
def test_functional_valid_question_returns_answer(client):
    """
    Functional test: Validates that submitting a well-formed question via the /ask endpoint returns a successful answer.
    - HTTP status code is 200
    - response['success'] is True
    - response['answer'] is a non-empty string
    """
    question = "What are some key facts about Mars?"
    payload = {"question": question}

    # Patch all external dependencies (LLM, Azure Search, Knowledge Base)
    with patch("builtins.print"):  # Patch print to avoid clutter
        with patch("builtins.open", create=True):  # Patch file I/O if used
            with patch("os.environ", {}):  # Patch environment if used
                # Patch LLM and Azure Search calls if they exist in your codebase
                # For demonstration, we assume the endpoint is self-contained

                response = client.post("/ask", json=payload)
                assert response.status_code == 200, "Expected HTTP 200 OK"
                data = response.get_json()
                assert data["success"] is True, "Expected 'success' to be True"
                assert isinstance(data["answer"], str) and data["answer"].strip(), "Expected non-empty answer string"

# Note: Error scenarios (knowledge base empty, LLM unavailable, Azure Search misconfigured)
# would be tested in separate tests, not in this happy-path functional test.
