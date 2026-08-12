"""Tests that guard against the 401 AuthenticationError production outage.

Root cause: an invalid/expired OPENAI_API_KEY caused every trace to fail at
the ChatOpenAI call inside Supervisor_Agent, crashing the whole LangGraph
workflow with an unhandled AuthenticationError.

These tests verify:
1. The process fails at startup (not at first request) when OPENAI_API_KEY is absent.
2. supervisor_agent returns a FINISH route instead of re-raising on LLM errors.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


def test_missing_openai_key_raises_at_startup():
    """_check_openai_key must raise EnvironmentError when key is absent."""
    from langgraph_fin_agent.graph import _check_openai_key

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            _check_openai_key()


def test_empty_openai_key_raises_at_startup():
    """_check_openai_key must raise EnvironmentError when key is an empty string."""
    from langgraph_fin_agent.graph import _check_openai_key

    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=True):
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            _check_openai_key()


def test_valid_openai_key_passes():
    """_check_openai_key must return the key when it is set."""
    from langgraph_fin_agent.graph import _check_openai_key

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        result = _check_openai_key()
    assert result == "sk-test-key"


def test_supervisor_agent_returns_finish_on_auth_error():
    """supervisor_agent must catch AuthenticationError and route to FINISH.

    Before this fix supervisor_agent had no try/except, so a 401 from OpenAI
    propagated through every LangGraph node and produced a total trace failure.
    """
    from openai import AuthenticationError as OpenAIAuthError
    from langchain_core.messages import HumanMessage

    import langgraph_fin_agent.graph as graph_module

    state = {"messages": [HumanMessage(content="hello")], "next": ""}

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = OpenAIAuthError(
        message="Incorrect API key provided",
        response=MagicMock(status_code=401, headers={}),
        body={"error": {"message": "Incorrect API key", "code": "invalid_api_key"}},
    )

    with patch.object(graph_module, "LLM") as mock_llm:
        mock_llm.with_structured_output.return_value = mock_chain
        result = graph_module.supervisor_agent(state)

    assert result.get("next") == "FINISH", (
        "supervisor_agent should route to FINISH on error, not propagate the exception"
    )
    messages = result.get("messages", [])
    assert messages, "supervisor_agent should include an error message in state"
    assert "error" in messages[-1].content.lower()
