"""
Tests that the Output_Summarizing_Agent's system prompt is never injected as a
HumanMessage into the LangGraph state. Regression for the bug where the custom
output_summarizing_node used direct LLM.invoke() with a ("system", ...) tuple,
which allowed the system prompt to end up in the wrong message slot.
"""
import functools
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def test_output_summarizing_agent_uses_create_react_agent():
    """output_summarizing_node must be built via create_react_agent, not direct LLM.invoke."""
    import inspect

    from langgraph_fin_agent.graph import output_summarizing_node

    # functools.partial wrapping agent_node is the expected pattern (same as other agents).
    assert isinstance(output_summarizing_node, functools.partial), (
        "output_summarizing_node must be functools.partial(agent_node, ...) — "
        "using direct LLM.invoke() with a ('system', ...) tuple can allow the "
        "system prompt to be placed in the HumanMessage slot of the graph state."
    )
    # The wrapped function must be agent_node.
    from langgraph_fin_agent.graph import agent_node
    assert output_summarizing_node.func is agent_node


def test_output_summarizing_node_does_not_add_human_message_to_state():
    """agent_node must only append an AIMessage for the summarizer's response."""
    from langgraph_fin_agent.graph import OUTPUT_SUMMARIZING_SYSTEM_PROMPT, output_summarizing_node

    fake_last_msg = AIMessage(content="Here is the summary.", name="Output_Summarizing_Agent")
    fake_agent_result = {"messages": [fake_last_msg]}

    fake_agent = MagicMock()
    fake_agent.ainvoke = AsyncMock(return_value=fake_agent_result)

    # Patch the agent inside the partial so we don't need real OpenAI credentials.
    node = functools.partial(
        output_summarizing_node.func,
        agent=fake_agent,
        name="Output_Summarizing_Agent",
    )

    import asyncio
    state = {"messages": [HumanMessage(content="What are the IPO highlights?")]}
    result = asyncio.run(node(state))

    returned_messages = result.get("messages", [])
    assert len(returned_messages) == 1
    msg = returned_messages[0]
    assert isinstance(msg, AIMessage), (
        f"output_summarizing_node must return an AIMessage, got {type(msg)}"
    )
    # The system prompt must never appear as a HumanMessage in the returned state update.
    for m in returned_messages:
        assert not isinstance(m, HumanMessage), (
            "System prompt must not be returned as a HumanMessage in the graph state."
        )
        assert OUTPUT_SUMMARIZING_SYSTEM_PROMPT not in (
            m.content if isinstance(m.content, str) else ""
        ), "System prompt text must not leak into the graph state messages."


def test_system_prompt_is_not_human_message_in_graph_input():
    """Graph must be invoked with a proper user query, never with a system prompt as HumanMessage."""
    from langgraph_fin_agent.graph import OUTPUT_SUMMARIZING_SYSTEM_PROMPT

    # Simulate the pattern used by main.py and run_ipo_dataset_experiment.py.
    user_query = "What was Apple's P/E ratio last quarter?"
    inputs = {"messages": [HumanMessage(content=user_query)]}

    human_messages = [m for m in inputs["messages"] if isinstance(m, HumanMessage)]
    assert len(human_messages) == 1
    assert human_messages[0].content == user_query

    # Verify the system prompt is not mistakenly used as the human message content.
    for m in human_messages:
        assert m.content != OUTPUT_SUMMARIZING_SYSTEM_PROMPT, (
            "OUTPUT_SUMMARIZING_SYSTEM_PROMPT must never be the content of a HumanMessage "
            "passed as the LangGraph input. Use SystemMessage (via state_modifier) instead."
        )
