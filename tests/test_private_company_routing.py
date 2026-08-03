"""Tests for private company routing fix.

Regression tests for the issue where Financial_Data_Agent was called for
private companies (e.g., SpaceX), causing it to invoke financial tools that
return incorrect or irrelevant results for unlisted companies.
"""
import pytest
from unittest.mock import patch

from langgraph_fin_agent.prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    FINANCIAL_DATA_SYSTEM_PROMPT,
)


class TestOrchestratorPromptPrivateCompanyGuidance:
    """Verify the Supervisor prompt explicitly restricts Financial_Data_Agent to public companies."""

    def test_financial_agent_restricted_to_public_companies(self):
        assert "publicly traded" in ORCHESTRATOR_SYSTEM_PROMPT.lower(), (
            "Supervisor prompt must restrict Financial_Data_Agent to publicly traded companies"
        )

    def test_private_company_routing_to_web_research(self):
        prompt_lower = ORCHESTRATOR_SYSTEM_PROMPT.lower()
        assert "private" in prompt_lower, (
            "Supervisor prompt must mention routing private companies to Web_Research_Agent"
        )
        assert "web_research_agent" in prompt_lower, (
            "Supervisor prompt must reference Web_Research_Agent as fallback for private companies"
        )

    def test_uncertainty_routing_preference(self):
        """When trading status is uncertain, Web_Research_Agent should be preferred."""
        prompt_lower = ORCHESTRATOR_SYSTEM_PROMPT.lower()
        assert "uncertain" in prompt_lower or "unsure" in prompt_lower, (
            "Supervisor prompt must specify Web_Research_Agent preference when trading status is uncertain"
        )

    def test_spacex_named_as_private_example(self):
        """SpaceX (the canonical failing case) should be named as an example of a private company."""
        assert "spacex" in ORCHESTRATOR_SYSTEM_PROMPT.lower(), (
            "Supervisor prompt must mention SpaceX as an example of a private company to avoid misrouting"
        )

    def test_financial_agent_section_mentions_public_restriction(self):
        fin_idx = ORCHESTRATOR_SYSTEM_PROMPT.find("Financial_Data_Agent")
        assert fin_idx != -1
        fin_section = ORCHESTRATOR_SYSTEM_PROMPT[fin_idx: fin_idx + 500].lower()
        assert "publicly traded" in fin_section or "public" in fin_section, (
            "Financial_Data_Agent description must mention its public-only restriction"
        )


class TestFinancialAgentPromptPrivateCompanyGuard:
    """Verify the Financial_Data_Agent prompt guards against private company tool calls."""

    def test_private_company_guard_present(self):
        assert "private" in FINANCIAL_DATA_SYSTEM_PROMPT.lower(), (
            "Financial_Data_Agent prompt must include a guard for private companies"
        )

    def test_do_not_call_tools_for_private_companies(self):
        prompt_lower = FINANCIAL_DATA_SYSTEM_PROMPT.lower()
        assert "do not call" in prompt_lower or "not call" in prompt_lower, (
            "Financial_Data_Agent prompt must instruct agent NOT to call tools for private companies"
        )

    def test_private_company_response_message(self):
        """The agent must provide a clear fallback message for private company queries."""
        prompt_lower = FINANCIAL_DATA_SYSTEM_PROMPT.lower()
        assert "not publicly traded" in prompt_lower or "financial market data is not available" in prompt_lower, (
            "Financial_Data_Agent prompt must specify a fallback message for private companies"
        )

    def test_spacex_named_as_private_example(self):
        assert "spacex" in FINANCIAL_DATA_SYSTEM_PROMPT.lower(), (
            "Financial_Data_Agent prompt must name SpaceX as a private company example"
        )


class TestFinancialToolsErrorHandling:
    """Verify financial tools return clean error dicts when the API returns no data."""

    def test_get_company_profile_empty_response(self):
        from langgraph_fin_agent.tools import get_company_profile

        with patch("langgraph_fin_agent.tools._fmp_request", return_value=[]):
            result = get_company_profile.invoke({"symbol": "SPACEX_PRIVATE"})
        assert "error" in result, "get_company_profile must return an error dict for empty API response"

    def test_get_stock_price_empty_response(self):
        from langgraph_fin_agent.tools import get_stock_price

        with patch("langgraph_fin_agent.tools._fmp_request", return_value=[]):
            result = get_stock_price.invoke({"symbol": "SPACEX_PRIVATE"})
        assert "error" in result, "get_stock_price must return an error dict for empty API response"

    def test_get_market_cap_empty_response(self):
        from langgraph_fin_agent.tools import get_market_cap

        with patch("langgraph_fin_agent.tools._fmp_request", return_value=[]):
            result = get_market_cap.invoke({"symbol": "SPACEX_PRIVATE"})
        assert "error" in result, "get_market_cap must return an error dict for empty API response"
