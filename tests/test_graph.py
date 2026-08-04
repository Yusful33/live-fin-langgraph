"""Tests for the graph module — specifically the system-prompt injection fixes."""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Must be set before graph.py is imported; graph.py creates ChatOpenAI at module level.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")

# Patch module-level side effects that require a live environment before importing.
_mock_react_agent = MagicMock()
patch("langgraph.prebuilt.create_react_agent", return_value=_mock_react_agent).start()
patch("langgraph_fin_agent.graph.create_react_agent", return_value=_mock_react_agent).start()

from langchain_core.messages import HumanMessage, AIMessage

from langgraph_fin_agent.graph import (
    validate_input_node,
    output_summarizing_node,
    OUTPUT_SUMMARIZING_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    FINANCIAL_DATA_SYSTEM_PROMPT,
    WEB_RESEARCH_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# validate_input_node — entry guard
# ---------------------------------------------------------------------------

def test_validate_input_node_passes_normal_query():
    state = {"messages": [HumanMessage(content="What is Apple's stock price?")], "next": ""}
    result = validate_input_node(state)
    assert result == {}


def test_validate_input_node_passes_empty_messages():
    state = {"messages": [], "next": ""}
    assert validate_input_node(state) == {}


def test_validate_input_node_passes_no_messages_key():
    assert validate_input_node({}) == {}


def test_validate_input_node_rejects_output_summarizing_prompt():
    """Regression: the exact prompt seen in production traces must be blocked."""
    state = {
        "messages": [HumanMessage(content=OUTPUT_SUMMARIZING_SYSTEM_PROMPT)],
        "next": "",
    }
    with pytest.raises(ValueError, match="agent system prompt"):
        validate_input_node(state)


def test_validate_input_node_rejects_orchestrator_prompt():
    state = {
        "messages": [HumanMessage(content=ORCHESTRATOR_SYSTEM_PROMPT)],
        "next": "",
    }
    with pytest.raises(ValueError, match="agent system prompt"):
        validate_input_node(state)


def test_validate_input_node_rejects_financial_data_prompt():
    state = {
        "messages": [HumanMessage(content=FINANCIAL_DATA_SYSTEM_PROMPT)],
        "next": "",
    }
    with pytest.raises(ValueError, match="agent system prompt"):
        validate_input_node(state)


def test_validate_input_node_rejects_web_research_prompt():
    state = {
        "messages": [HumanMessage(content=WEB_RESEARCH_SYSTEM_PROMPT)],
        "next": "",
    }
    with pytest.raises(ValueError, match="agent system prompt"):
        validate_input_node(state)


def test_validate_input_node_only_checks_first_human_message():
    """AIMessages carrying agent content between human turns must not trigger the guard."""
    state = {
        "messages": [
            HumanMessage(content="What is the market cap of TSLA?"),
            AIMessage(content=OUTPUT_SUMMARIZING_SYSTEM_PROMPT, name="Output_Summarizing_Agent"),
        ],
        "next": "",
    }
    # The first HumanMessage is a normal query — guard should pass.
    result = validate_input_node(state)
    assert result == {}


# ---------------------------------------------------------------------------
# output_summarizing_node — message role fix
# ---------------------------------------------------------------------------

@patch("langgraph_fin_agent.graph.LLM")
def test_output_summarizing_node_uses_human_role(mock_llm):
    """output_summarizing_node must send a 'human' turn to the LLM, not 'assistant'.

    Before the fix the second message used the 'assistant' role, which caused the
    model to receive [SystemMessage, AIMessage] with no user turn.  This unusual
    continuation pattern could lead the model to echo the system prompt as its
    response, which then bled into the next invocation's HumanMessage slot.
    """
    mock_response = MagicMock()
    mock_response.content = "TSLA market cap is approximately $800B."
    mock_llm.invoke.return_value = mock_response

    state = {
        "messages": [
            HumanMessage(content="What is Tesla's market cap?"),
            AIMessage(content="TSLA market cap: ~$800B", name="Financial_Data_Agent"),
        ],
        "next": "",
    }

    result = output_summarizing_node(state)

    call_args = mock_llm.invoke.call_args[0][0]
    assert len(call_args) == 2, "LLM should receive exactly two messages"
    role, _ = call_args[1]
    assert role == "human", (
        f"Second message to LLM must use 'human' role to give the model a proper user "
        f"turn; got {role!r}. Using 'assistant' role caused system-prompt echo."
    )

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "TSLA market cap is approximately $800B."
    assert result["messages"][0].name == "Output_Summarizing_Agent"


@patch("langgraph_fin_agent.graph.LLM")
def test_output_summarizing_node_includes_all_state_messages(mock_llm):
    mock_response = MagicMock()
    mock_response.content = "Summary."
    mock_llm.invoke.return_value = mock_response

    state = {
        "messages": [
            HumanMessage(content="Compare AAPL and MSFT P/E ratios"),
            AIMessage(content="AAPL P/E: 28, MSFT P/E: 35", name="Financial_Data_Agent"),
        ],
        "next": "",
    }

    output_summarizing_node(state)

    _, human_content = mock_llm.invoke.call_args[0][0][1]
    assert "AAPL P/E: 28" in human_content
    assert "Compare AAPL and MSFT" in human_content
