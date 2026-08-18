"""Tests for HTTP error handling in financial data tools."""
import json
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

from langgraph_fin_agent.tools import get_financial_ratios, get_key_metrics


def _make_http_error(code: int, reason: str) -> HTTPError:
    return HTTPError(
        url="https://example.com",
        code=code,
        msg=reason,
        hdrs=None,  # type: ignore[arg-type]
        fp=BytesIO(b""),
    )


def test_get_financial_ratios_returns_error_list_on_402():
    """get_financial_ratios must wrap a 402 error dict in a list, not return a bare dict."""
    with patch("langgraph_fin_agent.tools.urlopen") as mock_urlopen, \
         patch("langgraph_fin_agent.tools.os.getenv", return_value="dummy_key"):
        mock_urlopen.side_effect = _make_http_error(402, "Payment Required")
        result = get_financial_ratios.invoke({"symbol": "AAPL", "period": "quarter"})

    assert isinstance(result, list), "Result must be a list to match the declared return type"
    assert len(result) == 1
    assert "error" in result[0]
    assert "402" in result[0]["error"]


def test_get_key_metrics_returns_error_list_on_402():
    """get_key_metrics must wrap a 402 error dict in a list, not return a bare dict."""
    with patch("langgraph_fin_agent.tools.urlopen") as mock_urlopen, \
         patch("langgraph_fin_agent.tools.os.getenv", return_value="dummy_key"):
        mock_urlopen.side_effect = _make_http_error(402, "Payment Required")
        result = get_key_metrics.invoke({"symbol": "MSFT", "period": "quarter"})

    assert isinstance(result, list), "Result must be a list to match the declared return type"
    assert len(result) == 1
    assert "error" in result[0]
    assert "402" in result[0]["error"]


def test_get_financial_ratios_returns_error_list_on_other_http_errors():
    """get_financial_ratios wraps any HTTP error in a list."""
    with patch("langgraph_fin_agent.tools.urlopen") as mock_urlopen, \
         patch("langgraph_fin_agent.tools.os.getenv", return_value="dummy_key"):
        mock_urlopen.side_effect = _make_http_error(500, "Internal Server Error")
        result = get_financial_ratios.invoke({"symbol": "AAPL", "period": "annual"})

    assert isinstance(result, list)
    assert "error" in result[0]


def test_get_key_metrics_returns_data_on_success():
    """get_key_metrics returns the API data unchanged on success."""
    fake_data = [{"symbol": "AAPL", "marketCap": 3000000000000}]

    class FakeResponse:
        def read(self):
            return json.dumps(fake_data).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    with patch("langgraph_fin_agent.tools.urlopen", return_value=FakeResponse()), \
         patch("langgraph_fin_agent.tools.os.getenv", return_value="dummy_key"):
        result = get_key_metrics.invoke({"symbol": "AAPL", "period": "quarter"})

    assert result == fake_data
