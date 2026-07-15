"""Regression tests for looking up recently IPO'd stocks.

The bug: users asking about companies that IPO'd after the LLM's knowledge
cutoff (e.g. Circle/CRCL, CoreWeave/CRWV, Chime/CHYM) got wrong or empty
answers because:

1. Every financial-data tool required a ticker symbol as input, and
2. There was no tool for looking up a ticker from a company name.

The LLM was forced to guess a ticker, and for post-cutoff IPOs it either
returned a stale ticker (belonging to a different company that once used
the symbol), returned "no data available", or refused to answer.

The fix adds a `search_symbol` tool that queries FMP's `/search` endpoint
so the agent can resolve unfamiliar company names to their current ticker
before calling the other financial-data tools.
"""

from unittest.mock import patch

from langgraph_fin_agent import tools


def _fmp_response(company: str):
    """Canned FMP `/search` results for a handful of recent IPOs."""
    responses = {
        "circle": [
            {
                "symbol": "CRCL",
                "name": "Circle Internet Group, Inc.",
                "currency": "USD",
                "stockExchange": "New York Stock Exchange",
                "exchangeShortName": "NYSE",
            }
        ],
        "coreweave": [
            {
                "symbol": "CRWV",
                "name": "CoreWeave, Inc.",
                "currency": "USD",
                "stockExchange": "NASDAQ",
                "exchangeShortName": "NASDAQ",
            }
        ],
        "chime": [
            {
                "symbol": "CHYM",
                "name": "Chime Financial, Inc.",
                "currency": "USD",
                "stockExchange": "NASDAQ",
                "exchangeShortName": "NASDAQ",
            }
        ],
        "reddit": [
            {
                "symbol": "RDDT",
                "name": "Reddit, Inc.",
                "currency": "USD",
                "stockExchange": "New York Stock Exchange",
                "exchangeShortName": "NYSE",
            }
        ],
        "spacex": [],
    }
    return responses.get(company.lower(), [])


def test_search_symbol_tool_is_exported():
    """The new tool must be exposed as a LangChain tool."""
    assert hasattr(tools, "search_symbol"), "search_symbol tool missing"


def test_financial_data_agent_has_search_tool():
    """The Financial_Data_Agent's toolset must include search_symbol."""
    from langgraph_fin_agent.graph import FINANCIAL_DATA_TOOLS

    names = {t.name for t in FINANCIAL_DATA_TOOLS}
    assert "search_symbol" in names, (
        "search_symbol must be registered on the financial data agent so it "
        "can look up tickers for recently IPO'd companies"
    )


def test_search_symbol_resolves_recent_ipo():
    """search_symbol should return the current ticker for a recent IPO."""
    with patch.object(
        tools, "_fmp_request", side_effect=lambda ep, params=None: _fmp_response(params["query"])
    ):
        result = tools.search_symbol.invoke({"query": "Circle"})

    assert isinstance(result, list) and result, "expected non-empty results"
    assert result[0]["symbol"] == "CRCL"


def test_search_symbol_resolves_multiple_recent_ipos():
    cases = {
        "CoreWeave": "CRWV",
        "Chime": "CHYM",
        "Reddit": "RDDT",
    }
    for company, expected_ticker in cases.items():
        with patch.object(
            tools,
            "_fmp_request",
            side_effect=lambda ep, params=None: _fmp_response(params["query"]),
        ):
            result = tools.search_symbol.invoke({"query": company})
        assert result and result[0]["symbol"] == expected_ticker, (
            f"expected ticker {expected_ticker} for {company}, got {result}"
        )


def test_search_symbol_returns_empty_for_private_company():
    """Private companies (e.g. SpaceX) should return an empty list, not
    a hallucinated ticker."""
    with patch.object(
        tools, "_fmp_request", side_effect=lambda ep, params=None: _fmp_response(params["query"])
    ):
        result = tools.search_symbol.invoke({"query": "SpaceX"})

    assert result == [], "private companies must return an empty result set"


def test_search_symbol_propagates_api_errors():
    """When FMP returns an error we must surface it, not silently drop it."""
    err = {"error": "API access forbidden. Please check your API key."}
    with patch.object(tools, "_fmp_request", return_value=err):
        result = tools.search_symbol.invoke({"query": "Circle"})

    assert result == [err]


def test_orchestrator_prompt_mentions_symbol_lookup():
    """The supervisor prompt should teach the agent to use search_symbol
    when the ticker for a company is not known — otherwise the agent may
    still guess for recently IPO'd companies."""
    from langgraph_fin_agent.graph import FINANCIAL_DATA_SYSTEM_PROMPT

    text = FINANCIAL_DATA_SYSTEM_PROMPT.lower()
    assert "search_symbol" in text, (
        "the financial data agent's system prompt must reference the "
        "search_symbol tool so it uses it before falling back to a guess"
    )
