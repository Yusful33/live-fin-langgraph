"""
Tests that the Supervisor and Financial_Data_Agent prompts correctly handle
private (non-publicly-traded) company queries, preventing futile tool calls.

These tests only validate prompt content; they do not call any LLM or API.
"""

import os
import sys
import types
import unittest.mock as mock

# Stub out heavy dependencies so the module can be imported without
# a real OpenAI key or a matching langgraph install.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")

_dummy_agent = mock.MagicMock()


def _stub_create_react_agent(*args, **kwargs):
    return _dummy_agent


# Patch langchain_openai.ChatOpenAI and langgraph create_react_agent
# before the graph module is imported.
with mock.patch("langchain_openai.ChatOpenAI", return_value=mock.MagicMock()):
    with mock.patch(
        "langgraph.prebuilt.create_react_agent", side_effect=_stub_create_react_agent
    ):
        # Ensure a clean import even if graph was already imported
        for mod in list(sys.modules):
            if "langgraph_fin_agent" in mod:
                del sys.modules[mod]

        from langgraph_fin_agent.graph import (  # noqa: E402
            ORCHESTRATOR_SYSTEM_PROMPT,
            FINANCIAL_DATA_SYSTEM_PROMPT,
        )


class TestOrchestratorPrompt:
    """Supervisor prompt must steer private-company queries to Web_Research_Agent."""

    def test_orchestrator_mentions_publicly_traded_restriction(self):
        lower = ORCHESTRATOR_SYSTEM_PROMPT.lower()
        assert "publicly traded" in lower, (
            "Supervisor prompt must mention 'publicly traded' to restrict Financial_Data_Agent use"
        )

    def test_orchestrator_names_private_company_example(self):
        # SpaceX is the canonical example from the production failure
        assert "SpaceX" in ORCHESTRATOR_SYSTEM_PROMPT, (
            "Supervisor prompt should name SpaceX as an example private company"
        )

    def test_orchestrator_routes_private_to_web_research(self):
        lower = ORCHESTRATOR_SYSTEM_PROMPT.lower()
        assert "private" in lower, "Supervisor prompt must mention 'private' companies"
        assert "web_research_agent" in lower, (
            "Supervisor prompt must explicitly mention routing to Web_Research_Agent for private companies"
        )

    def test_orchestrator_clarifies_financial_agent_limitation(self):
        fin_idx = ORCHESTRATOR_SYSTEM_PROMPT.find("Financial_Data_Agent")
        assert fin_idx != -1
        # The description of Financial_Data_Agent must mention it only handles public companies
        fin_section = ORCHESTRATOR_SYSTEM_PROMPT[fin_idx: fin_idx + 500]
        lower = fin_section.lower()
        assert "publicly traded" in lower or "public" in lower, (
            "Financial_Data_Agent description must mention its public-only restriction"
        )


class TestFinancialDataAgentPrompt:
    """Financial_Data_Agent prompt must guard against private-company tool calls."""

    def test_prompt_warns_about_private_companies(self):
        assert "private" in FINANCIAL_DATA_SYSTEM_PROMPT.lower(), (
            "Financial_Data_Agent prompt must warn about private companies"
        )

    def test_prompt_instructs_not_to_call_tools_for_private_companies(self):
        lower = FINANCIAL_DATA_SYSTEM_PROMPT.lower()
        assert "not publicly traded" in lower or "private" in lower, (
            "Prompt must instruct the agent not to call tools for private companies"
        )

    def test_prompt_provides_fallback_message(self):
        lower = FINANCIAL_DATA_SYSTEM_PROMPT.lower()
        assert "not publicly traded" in lower or "financial market data is not available" in lower, (
            "Prompt must provide a clear fallback message for private company queries"
        )

    def test_prompt_names_spacex_as_private_example(self):
        assert "SpaceX" in FINANCIAL_DATA_SYSTEM_PROMPT, (
            "Financial_Data_Agent prompt should name SpaceX as an example private company"
        )
