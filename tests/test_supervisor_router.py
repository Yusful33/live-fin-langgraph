"""Tests for the supervisor_router guard.

Verifies that the router prevents premature routing to Output_Summarizing_Agent
when no data-gathering agent has yet produced output.
"""
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_fin_agent.graph import supervisor_router


def _state(messages, next_agent):
    return {"messages": messages, "next": next_agent}


class TestSupervisorRouterGuard:
    def test_routes_to_financial_data_agent_when_no_data_and_supervisor_picks_summarizer(self):
        """Guard must redirect to Financial_Data_Agent when supervisor prematurely picks Output_Summarizing_Agent."""
        state = _state(
            messages=[HumanMessage(content="What is the stock price of HSBC?")],
            next_agent="Output_Summarizing_Agent",
        )
        assert supervisor_router(state) == "Financial_Data_Agent"

    def test_routes_to_financial_data_agent_when_no_data_and_supervisor_picks_finish(self):
        """Guard must redirect to Financial_Data_Agent when supervisor prematurely selects FINISH."""
        state = _state(
            messages=[HumanMessage(content="Should I invest in Apple stock?")],
            next_agent="FINISH",
        )
        assert supervisor_router(state) == "Financial_Data_Agent"

    def test_allows_summarizer_after_financial_data_agent_has_run(self):
        """Routing to Output_Summarizing_Agent is allowed once Financial_Data_Agent has produced output."""
        state = _state(
            messages=[
                HumanMessage(content="What is the stock price of HSBC?"),
                AIMessage(content='{"symbol": "HSBC", "price": 42.0}', name="Financial_Data_Agent"),
            ],
            next_agent="Output_Summarizing_Agent",
        )
        assert supervisor_router(state) == "Output_Summarizing_Agent"

    def test_allows_finish_after_web_research_agent_has_run(self):
        """FINISH is valid once Web_Research_Agent has produced output."""
        state = _state(
            messages=[
                HumanMessage(content="Tell me about Tesla's latest news."),
                AIMessage(content="Tesla Q3 results announced...", name="Web_Research_Agent"),
            ],
            next_agent="FINISH",
        )
        assert supervisor_router(state) == "FINISH"

    def test_passes_through_financial_data_agent_routing_unchanged(self):
        """Routing to Financial_Data_Agent should always pass through unmodified."""
        state = _state(
            messages=[HumanMessage(content="Get me AAPL financials.")],
            next_agent="Financial_Data_Agent",
        )
        assert supervisor_router(state) == "Financial_Data_Agent"

    def test_passes_through_web_research_agent_routing_unchanged(self):
        """Routing to Web_Research_Agent should always pass through unmodified."""
        state = _state(
            messages=[HumanMessage(content="Get news from example.com.")],
            next_agent="Web_Research_Agent",
        )
        assert supervisor_router(state) == "Web_Research_Agent"

    def test_allows_summarizer_after_both_agents_have_run(self):
        """Routing to Output_Summarizing_Agent is fine when multiple data agents ran."""
        state = _state(
            messages=[
                HumanMessage(content="Tell me about AAPL's latest news and stock price."),
                AIMessage(content='{"price": 310.03}', name="Financial_Data_Agent"),
                AIMessage(content="Apple Q4 results released.", name="Web_Research_Agent"),
            ],
            next_agent="Output_Summarizing_Agent",
        )
        assert supervisor_router(state) == "Output_Summarizing_Agent"
