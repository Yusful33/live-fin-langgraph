"""Tests for tool response truncation to prevent high LLM synthesis latency.

The Financial_Data_Agent was passing full untruncated tool responses (9591 prompt
tokens) into its synthesis ChatOpenAI call, causing 21.5 s latency.  These tests
verify that _truncate_response keeps individual tool payloads within budget.
"""

import json
import pytest
from unittest.mock import patch

from langgraph_fin_agent.tools import (
    _truncate_response,
    MAX_TOOL_RESPONSE_CHARS,
    _MAX_FIELD_CHARS,
    get_company_profile,
    get_financial_ratios,
    get_key_metrics,
    generate_single_line_item_query,
)


# ---------------------------------------------------------------------------
# _truncate_response unit tests
# ---------------------------------------------------------------------------

def test_truncate_list_keeps_only_first_entry():
    many_periods = [{"year": i, "ratio": i * 1.1} for i in range(10)]
    result = _truncate_response(many_periods)
    assert result == [{"year": 0, "ratio": 0.0}]


def test_truncate_dict_shortens_long_string_fields():
    long_desc = "x" * 500
    data = {"name": "ACME Corp", "description": long_desc, "price": 42.0}
    result = _truncate_response(data)
    assert isinstance(result, dict)
    assert result["name"] == "ACME Corp"
    assert result["price"] == 42.0
    assert len(result["description"]) == _MAX_FIELD_CHARS + 3  # "..." suffix
    assert result["description"].endswith("...")


def test_truncate_serialized_output_within_budget():
    """Whatever _truncate_response returns must serialize within budget."""
    # Build a dict that is large even after field-level truncation
    data = {f"field_{i}": "y" * 300 for i in range(20)}
    result = _truncate_response(data)
    serialized = json.dumps(result) if not isinstance(result, str) else result
    assert len(serialized) <= MAX_TOOL_RESPONSE_CHARS + 3  # +3 for "..." suffix


def test_truncate_small_payload_unchanged():
    small = {"symbol": "AAPL", "price": 180.5}
    result = _truncate_response(small)
    assert result == small


def test_truncate_empty_list_unchanged():
    result = _truncate_response([])
    assert result == []


# ---------------------------------------------------------------------------
# Tool integration tests (mocked API layer)
# ---------------------------------------------------------------------------

SYNTHETIC_PROFILE = {
    "symbol": "FAKE",
    "companyName": "Fake Corp",
    "description": "A" * 600,  # deliberately verbose description
    "industry": "Technology",
    "price": 99.99,
    "marketCap": 1_000_000_000,
}

SYNTHETIC_RATIOS = [
    {"symbol": "FAKE", "date": f"202{y}-12-31", "peRatio": 25.0 + y, "debtEquityRatio": 0.5}
    for y in range(5)
]

SYNTHETIC_METRICS = [
    {"symbol": "FAKE", "date": f"202{y}-12-31", "revenuePerShare": 10.0 + y, "roe": 0.2}
    for y in range(5)
]

SYNTHETIC_INCOME = [
    {"symbol": "FAKE", "date": f"202{y}-12-31", "revenue": 1_000_000 * (y + 1)}
    for y in range(5)
]


def test_get_company_profile_truncates_description():
    with patch("langgraph_fin_agent.tools._fmp_request", return_value=[SYNTHETIC_PROFILE]):
        result = get_company_profile.invoke({"symbol": "FAKE"})
    assert isinstance(result, dict)
    assert result["companyName"] == "Fake Corp"
    assert len(result["description"]) <= _MAX_FIELD_CHARS + 3


def test_get_financial_ratios_returns_single_period():
    with patch("langgraph_fin_agent.tools._fmp_request", return_value=SYNTHETIC_RATIOS):
        result = get_financial_ratios.invoke({"symbol": "FAKE"})
    # Should be a list with only the most recent entry
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["date"] == "2020-12-31"


def test_get_key_metrics_returns_single_period():
    with patch("langgraph_fin_agent.tools._fmp_request", return_value=SYNTHETIC_METRICS):
        result = get_key_metrics.invoke({"symbol": "FAKE"})
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["date"] == "2020-12-31"


def test_generate_single_line_item_query_returns_single_period():
    with patch("langgraph_fin_agent.tools._fmp_request", return_value=SYNTHETIC_INCOME):
        result = generate_single_line_item_query.invoke(
            {"ticker": "FAKE", "statement": "income-statement", "period": "annual"}
        )
    assert isinstance(result, list)
    assert len(result) == 1


def test_tool_response_fits_within_budget():
    """End-to-end: a realistic multi-tool scenario stays within per-tool budget."""
    with patch("langgraph_fin_agent.tools._fmp_request", return_value=[SYNTHETIC_PROFILE]):
        profile = get_company_profile.invoke({"symbol": "FAKE"})
    with patch("langgraph_fin_agent.tools._fmp_request", return_value=SYNTHETIC_RATIOS):
        ratios = get_financial_ratios.invoke({"symbol": "FAKE"})
    with patch("langgraph_fin_agent.tools._fmp_request", return_value=SYNTHETIC_METRICS):
        metrics = get_key_metrics.invoke({"symbol": "FAKE"})

    for label, payload in [("profile", profile), ("ratios", ratios), ("metrics", metrics)]:
        size = len(json.dumps(payload))
        assert size <= MAX_TOOL_RESPONSE_CHARS + 3, (
            f"{label} response ({size} chars) exceeds MAX_TOOL_RESPONSE_CHARS={MAX_TOOL_RESPONSE_CHARS}"
        )


def test_error_passthrough_not_truncated():
    """Error dicts from _fmp_request must be returned unchanged (no masking)."""
    error_response = {"error": "API access forbidden. Please check your API key."}
    with patch("langgraph_fin_agent.tools._fmp_request", return_value=error_response):
        result = get_financial_ratios.invoke({"symbol": "FAKE"})
    assert result == error_response
