
import pytest
from unittest.mock import patch, MagicMock
import logging

# Assume the following imports are from the agent module under test
# from agent import Agent, RetrievalClient, LLMClient

@pytest.fixture
def fallback_response():
    return "Sorry, I am unable to answer your question at the moment."

@pytest.fixture
def agent(fallback_response):
    # Mocked dependencies for the Agent
    retrieval_client = MagicMock()
    llm_client = MagicMock()
    # The Agent class should accept retrieval_client, llm_client, and fallback_response as init args
    # If your Agent signature differs, adjust accordingly
    agent = Agent(
        retrieval_client=retrieval_client,
        llm_client=llm_client,
        fallback_response=fallback_response
    )
    return agent

@pytest.fixture
def question():
    return "What is the capital of France?"

def _simulate_retrieval_failure(*args, **kwargs):
    raise Exception("Retrieval failed")

def _simulate_llm_failure(*args, **kwargs):
    raise Exception("LLM failed")

@pytest.mark.integration
def test_integration_error_handling_and_fallback_response_retrieval_failure(agent, fallback_response, question, caplog):
    """
    Integration test: Checks that when RetrievalClient.retrieve_chunks fails,
    the fallback response is returned and error is logged.
    """
    # Patch the retrieval_client.retrieve_chunks to raise an exception
    agent.retrieval_client.retrieve_chunks.side_effect = _simulate_retrieval_failure
    # LLM should not be called, but if it is, return a dummy value
    agent.llm_client.generate_answer.return_value = "Should not be called"

    with caplog.at_level(logging.ERROR):
        answer = agent.process_question(question)

    assert answer == fallback_response, "Agent should return fallback response on retrieval failure"
    assert any("Retrieval failed" in record.message for record in caplog.records), "Error should be logged at error level"

@pytest.mark.integration
def test_integration_error_handling_and_fallback_response_llm_failure(agent, fallback_response, question, caplog):
    """
    Integration test: Checks that when LLMClient.generate_answer fails,
    the fallback response is returned and error is logged.
    """
    # Retrieval succeeds, but LLM fails
    agent.retrieval_client.retrieve_chunks.return_value = ["Paris is the capital of France."]
    agent.llm_client.generate_answer.side_effect = _simulate_llm_failure

    with caplog.at_level(logging.ERROR):
        answer = agent.process_question(question)

    assert answer == fallback_response, "Agent should return fallback response on LLM failure"
    assert any("LLM failed" in record.message for record in caplog.records), "Error should be logged at error level"
