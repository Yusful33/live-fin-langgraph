"""
Tests for out-of-domain (OOD) request handling in the supervisor agent.

Verifies that insurance claim inputs (and similar non-financial queries) are
declined gracefully rather than silently echoing the user's input verbatim.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage

# Set a dummy key before importing the module so the module-level LLM
# instantiation does not fail with a missing-key error.
os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")

from langgraph_fin_agent.graph import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    OUTPUT_SUMMARIZING_SYSTEM_PROMPT,
    output_summarizing_node,
    build_workflow,
    conditional_map,
)


class TestOrchestratorPromptOODGuidance:
    """The supervisor prompt must contain explicit OOD handling instructions."""

    def test_prompt_mentions_out_of_domain(self):
        assert "out-of-domain" in ORCHESTRATOR_SYSTEM_PROMPT.lower() or \
               "out of domain" in ORCHESTRATOR_SYSTEM_PROMPT.lower()

    def test_prompt_mentions_insurance(self):
        assert "insurance" in ORCHESTRATOR_SYSTEM_PROMPT.lower()

    def test_prompt_instructs_output_agent_for_ood(self):
        # Supervisor must route OOD requests to Output_Summarizing_Agent, not FINISH
        assert "Output_Summarizing_Agent" in ORCHESTRATOR_SYSTEM_PROMPT
        lower = ORCHESTRATOR_SYSTEM_PROMPT.lower()
        assert "finish" not in lower.split("out-of-domain")[1] or \
               "do not route to finish" in lower or \
               "not route to finish" in lower


class TestOutputSummarizingPromptOODGuidance:
    """The output agent prompt must tell the LLM how to handle OOD cases."""

    def test_prompt_addresses_ood_case(self):
        lower = OUTPUT_SUMMARIZING_SYSTEM_PROMPT.lower()
        assert "out-of-domain" in lower or "outside" in lower or "not able to assist" in lower

    def test_prompt_mentions_decline(self):
        lower = OUTPUT_SUMMARIZING_SYSTEM_PROMPT.lower()
        # Should instruct the agent to decline rather than echo/summarise nothing
        assert "decline" in lower or "not able" in lower or "cannot assist" in lower or \
               "polite" in lower


class TestFinishRoutesToOutputAgent:
    """FINISH in the conditional_map must route to Output_Summarizing_Agent, not END."""

    def test_finish_routes_to_output_summarizing_agent(self):
        assert conditional_map["FINISH"] == "Output_Summarizing_Agent"


class TestOutputSummarizingNodeOOD:
    """output_summarizing_node must call the LLM even for OOD inputs."""

    def test_ood_input_calls_llm_and_returns_ai_message(self):
        """With only a human message in state (no financial data), the node should
        still invoke the LLM and return an AIMessage — not echo the input."""
        state = {
            "messages": [
                HumanMessage(content="HEALTH CLAIM | Policy #HL-99999, emergency surgery, $50k billed.")
            ],
            "next": "FINISH",
        }

        mock_response = MagicMock()
        mock_response.content = (
            "I'm sorry, but this assistant specialises in financial market data "
            "and is unable to assist with insurance claims. Please contact your "
            "insurance provider directly."
        )

        with patch("langgraph_fin_agent.graph.LLM") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = output_summarizing_node(state)

        # Must have called the LLM
        mock_llm.invoke.assert_called_once()

        # Output must be an AIMessage (not verbatim user input)
        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert msg.content != state["messages"][0].content, \
            "Output must not echo the user's input verbatim"
        assert msg.name == "Output_Summarizing_Agent"

    def test_ood_llm_prompt_includes_state_messages(self):
        """The LLM prompt passed by output_summarizing_node must include the
        user's original message so the model can reference the query type."""
        state = {
            "messages": [
                HumanMessage(content="AUTO CLAIM | Policy #AU-12345, rear-end collision.")
            ],
            "next": "FINISH",
        }

        mock_response = MagicMock()
        mock_response.content = "This assistant handles financial data only."

        with patch("langgraph_fin_agent.graph.LLM") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            output_summarizing_node(state)

        call_args = mock_llm.invoke.call_args[0][0]
        # The user's message content must appear somewhere in the prompt
        combined = " ".join(str(m) for m in call_args)
        assert "AUTO CLAIM" in combined


class TestWorkflowHasOutputSummarizingNode:
    """The compiled workflow graph must include the Output_Summarizing_Agent node."""

    def test_output_summarizing_agent_node_present(self):
        workflow = build_workflow()
        assert "Output_Summarizing_Agent" in workflow.nodes
