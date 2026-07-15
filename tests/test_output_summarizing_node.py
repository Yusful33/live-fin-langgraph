"""Tests for output_summarizing_node date injection.

Verifies that the Output_Summarizing_Agent injects the real current date into
its system prompt so the LLM cannot fall back to its training-data cutoff year.
"""
import sys
import types
from datetime import date
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Build and register stub modules BEFORE any import of the project package.
# ---------------------------------------------------------------------------

def _register(name, attrs=None, is_pkg=False):
    mod = types.ModuleType(name)
    mod.__spec__ = None
    if is_pkg:
        mod.__path__ = []
        mod.__package__ = name
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Clear any prior project imports
for _k in list(sys.modules):
    if _k.startswith("langgraph_fin_agent"):
        del sys.modules[_k]

# Minimal stub for dotenv
_register("dotenv", {"load_dotenv": lambda: None})

# langchain_core and sub-modules
_register("langchain_core", is_pkg=True)

class _AIMessage:
    def __init__(self, content, name=None, **kwargs):
        self.content = content
        self.name = name

class _HumanMessage:
    def __init__(self, content, **kwargs):
        self.content = content

_register("langchain_core.messages", {
    "AIMessage": _AIMessage,
    "HumanMessage": _HumanMessage,
    "BaseMessage": object,
})
_register("langchain_core.prompts", {
    "ChatPromptTemplate": MagicMock(),
    "MessagesPlaceholder": MagicMock(),
})
_register("langchain_core.tools", {
    "tool": lambda f=None, **kw: (f if f else lambda g: g),
    "StructuredTool": MagicMock(),
})

# langchain_openai
_mock_llm = MagicMock()
_register("langchain_openai", {"ChatOpenAI": MagicMock(return_value=_mock_llm)})

# langgraph
_register("langgraph", is_pkg=True)
_register("langgraph.graph", {"StateGraph": MagicMock(), "END": "END", "START": "START"})
_register("langgraph.prebuilt", {"create_react_agent": MagicMock(return_value=MagicMock())})
_register("langgraph.checkpoint", is_pkg=True)
_register("langgraph.checkpoint.memory", {"MemorySaver": MagicMock()})

# pydantic
_register("pydantic", {"BaseModel": object}, is_pkg=True)

# bs4 / requests
_register("bs4", {"BeautifulSoup": MagicMock()})
_register("requests", {})

# ---------------------------------------------------------------------------
# Now import the module under test.
# ---------------------------------------------------------------------------
sys.path.insert(0, "/workspace/live-fin-langgraph")

from langgraph_fin_agent.graph import (  # noqa: E402
    OUTPUT_SUMMARIZING_SYSTEM_PROMPT,
    output_summarizing_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(*contents):
    return {"messages": [_HumanMessage(c) for c in contents], "next": ""}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_prompt_template_has_date_placeholders():
    """OUTPUT_SUMMARIZING_SYSTEM_PROMPT must contain {current_date} and {current_year}."""
    assert "{current_date}" in OUTPUT_SUMMARIZING_SYSTEM_PROMPT, (
        "Prompt template must include {current_date} placeholder"
    )
    assert "{current_year}" in OUTPUT_SUMMARIZING_SYSTEM_PROMPT, (
        "Prompt template must include {current_year} placeholder"
    )


def test_prompt_template_bans_fabrication():
    """Prompt must explicitly instruct the LLM not to invent dates."""
    lower = OUTPUT_SUMMARIZING_SYSTEM_PROMPT.lower()
    assert "never" in lower or "do not" in lower or "fabricat" in lower, (
        "Prompt must tell the LLM not to fabricate or invent dates"
    )


def test_node_injects_current_year_into_system_prompt():
    """output_summarizing_node must inject the real current year, not a stale one."""
    captured = {}

    def fake_invoke(messages):
        captured["messages"] = messages
        result = MagicMock()
        result.content = "summary"
        return result

    _mock_llm.invoke.side_effect = fake_invoke

    output_summarizing_node(_state("What is NVIDIA's stock today?"))

    system_content = captured["messages"][0][1]
    today = date.today()
    assert str(today.year) in system_content, (
        f"System prompt must contain current year {today.year}; "
        f"got:\n{system_content}"
    )
    # The stale year observed in production failures must not be hard-coded.
    assert "2023" not in system_content, (
        "System prompt must not hard-code stale year 2023"
    )


def test_node_injects_current_date_string():
    """output_summarizing_node must inject the human-readable current date."""
    captured = {}

    def fake_invoke(messages):
        captured["messages"] = messages
        result = MagicMock()
        result.content = "summary"
        return result

    _mock_llm.invoke.side_effect = fake_invoke

    output_summarizing_node(_state("How old is Apple Inc.?"))

    system_content = captured["messages"][0][1]
    today = date.today()
    expected = today.strftime("%B %d, %Y")
    assert expected in system_content, (
        f"System prompt must contain today's date '{expected}'; "
        f"got:\n{system_content}"
    )


def test_node_returns_ai_message_named_correctly():
    """output_summarizing_node must return an AIMessage named 'Output_Summarizing_Agent'."""
    result_mock = MagicMock()
    result_mock.content = "the final summary"
    _mock_llm.invoke.return_value = result_mock
    _mock_llm.invoke.side_effect = None

    result = output_summarizing_node(_state("Summarize this"))

    assert "messages" in result
    msg = result["messages"][0]
    assert isinstance(msg, _AIMessage)
    assert msg.name == "Output_Summarizing_Agent"
    assert msg.content == "the final summary"
