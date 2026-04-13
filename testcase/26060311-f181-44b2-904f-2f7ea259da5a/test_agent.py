
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask, jsonify, request

@pytest.fixture
def app():
    """
    Provides a Flask app instance configured for testing.
    """
    app = Flask(__name__)

    @app.route('/ask', methods=['POST'])
    def ask():
        try:
            data = request.get_json(force=True)
        except Exception:
            return jsonify({
                "success": False,
                "error": "Malformed JSON.",
                "details": {"json": "Invalid or malformed JSON payload."}
            }), 400

        question = data.get("question") if data else None
        if question is None:
            return jsonify({
                "success": False,
                "error": "Input validation failed.",
                "details": {"question": "This field is required."}
            }), 422
        if not isinstance(question, str) or not question.strip():
            return jsonify({
                "success": False,
                "error": "Input validation failed.",
                "details": {"question": "Question must not be empty."}
            }), 422

        # Simulate normal processing (not reached in these tests)
        return jsonify({"success": True, "answer": "42"}), 200

    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    """
    Provides a Flask test client.
    """
    return app.test_client()

def test_functional_input_validation_failure(client):
    """
    Ensures that submitting an empty question triggers input validation and returns a 422 error with appropriate details.
    """
    # Input: POST /ask with JSON body: {"question": "   "}
    response = client.post('/ask', json={"question": "   "})
    assert response.status_code == 422, "Expected HTTP 422 for empty question"
    data = response.get_json()
    assert data["success"] is False, "Expected success=False"
    assert "Input validation failed." in data["error"], "Expected input validation error message"
    assert "question" in data["details"], "Expected validation details for 'question'"

def test_functional_input_validation_failure_missing_question(client):
    """
    Ensures that missing 'question' field triggers input validation and returns a 422 error with appropriate details.
    """
    response = client.post('/ask', json={})
    assert response.status_code == 422, "Expected HTTP 422 for missing question"
    data = response.get_json()
    assert data["success"] is False, "Expected success=False"
    assert "Input validation failed." in data["error"], "Expected input validation error message"
    assert "question" in data["details"], "Expected validation details for 'question'"

def test_functional_input_validation_failure_malformed_json(client):
    """
    Ensures that malformed JSON triggers a 400 error and returns appropriate error details.
    """
    # Send invalid JSON (e.g., missing closing brace)
    response = client.post('/ask', data='{"question": "What?"', content_type='application/json')
    assert response.status_code == 400, "Expected HTTP 400 for malformed JSON"
    data = response.get_json()
    assert data["success"] is False, "Expected success=False"
    assert "Malformed JSON" in data["error"], "Expected malformed JSON error message"
    assert "json" in data["details"], "Expected validation details for JSON"

