"""
Tests that conversation history is correctly threaded across separate build_app()
invocations when using the same SQLite checkpoint store and thread_id.

This is the regression test for: Follow-up queries lose prior conversation context
(trace c3b14aa76a64bab04451e28cc6a75d11 only contained the current turn's message).
"""

import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, AIMessage


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a temporary SQLite checkpoint database."""
    return str(tmp_path / "test_checkpoints.db")


def _make_supervisor_response(next_agent: str):
    """Return a mock RouteResponse-like object."""
    mock = MagicMock()
    mock.next = next_agent
    return mock


def _build_patched_app(db_path: str):
    """Build the app with all LLM calls mocked out."""
    # Patch LLM at module level before importing graph so the module-level
    # ChatOpenAI() call is intercepted.
    with patch("langgraph_fin_agent.graph.LLM") as mock_llm:
        mock_llm.with_structured_output.return_value.invoke = MagicMock(
            return_value=_make_supervisor_response("FINISH")
        )
        mock_llm.invoke = MagicMock(
            return_value=AIMessage(content="Summary response")
        )

        from langgraph_fin_agent.graph import build_app
        app = build_app(db_path=db_path)
    return app, mock_llm


class TestConversationPersistence:
    """Verify that SqliteSaver threads message history across invocations."""

    def test_messages_accumulate_across_separate_app_instances(self, tmp_db):
        """
        Two separate build_app() calls sharing the same SQLite DB and thread_id
        must accumulate messages — simulating a process-restart scenario.

        Before the fix, MemorySaver lost state on every new build_app() call,
        so the second invocation's supervisor saw only the current turn's message.
        """
        thread_id = "test-session-001"

        # --- First invocation (e.g. "tell me about spacex") ---
        with (
            patch("langgraph_fin_agent.graph.LLM") as mock_llm1,
            patch("langgraph_fin_agent.graph.financial_data_agent"),
            patch("langgraph_fin_agent.graph.web_research_agent"),
        ):
            mock_llm1.with_structured_output.return_value.invoke = MagicMock(
                return_value=_make_supervisor_response("FINISH")
            )
            mock_llm1.invoke = MagicMock(
                return_value=AIMessage(content="SpaceX is a rocket company.")
            )

            from langgraph_fin_agent.graph import build_app

            app1 = build_app(db_path=tmp_db)
            config = {"configurable": {"thread_id": thread_id}}
            inputs1 = {"messages": [HumanMessage(content="tell me about spacex")]}
            asyncio.run(
                app1.ainvoke(inputs1, config)
            )

        # --- Second invocation (e.g. "what about tesla") — NEW app instance ---
        # This simulates a process restart: fresh build_app() but same DB + thread_id.
        with (
            patch("langgraph_fin_agent.graph.LLM") as mock_llm2,
            patch("langgraph_fin_agent.graph.financial_data_agent"),
            patch("langgraph_fin_agent.graph.web_research_agent"),
        ):
            captured_states = []

            def capture_supervisor(state):
                captured_states.append(state)
                return _make_supervisor_response("FINISH")

            mock_llm2.with_structured_output.return_value.invoke = capture_supervisor
            mock_llm2.invoke = MagicMock(
                return_value=AIMessage(content="Tesla is an EV company.")
            )

            # Re-import to get a fresh build_app bound to the new mock LLM.
            import importlib
            import langgraph_fin_agent.graph as graph_module
            importlib.reload(graph_module)

            app2 = graph_module.build_app(db_path=tmp_db)
            config = {"configurable": {"thread_id": thread_id}}
            inputs2 = {"messages": [HumanMessage(content="what about tesla")]}
            asyncio.run(app2.ainvoke(inputs2, config))

        # The supervisor in the second invocation must see BOTH turns' messages.
        assert len(captured_states) > 0, "Supervisor was never called in second run"
        second_run_messages = captured_states[0]["messages"]
        contents = [m.content for m in second_run_messages]

        assert any("spacex" in c.lower() for c in contents), (
            "Second invocation's supervisor state is missing SpaceX history. "
            f"Messages seen: {contents}"
        )
        assert any("tesla" in c.lower() for c in contents), (
            "Second invocation's supervisor state is missing the Tesla query. "
            f"Messages seen: {contents}"
        )

    def test_separate_thread_ids_are_isolated(self, tmp_db):
        """
        Two different thread_ids sharing the same SQLite DB must not bleed
        messages into each other.
        """
        with (
            patch("langgraph_fin_agent.graph.LLM") as mock_llm,
            patch("langgraph_fin_agent.graph.financial_data_agent"),
            patch("langgraph_fin_agent.graph.web_research_agent"),
        ):
            mock_llm.with_structured_output.return_value.invoke = MagicMock(
                return_value=_make_supervisor_response("FINISH")
            )
            mock_llm.invoke = MagicMock(
                return_value=AIMessage(content="Some response")
            )

            from langgraph_fin_agent.graph import build_app

            app = build_app(db_path=tmp_db)
            config_a = {"configurable": {"thread_id": "user-a"}}
            config_b = {"configurable": {"thread_id": "user-b"}}

            asyncio.run(
                app.ainvoke(
                    {"messages": [HumanMessage(content="user-a private query")]},
                    config_a,
                )
            )

            captured_states = []

            def capture(state):
                captured_states.append(state)
                return _make_supervisor_response("FINISH")

            mock_llm.with_structured_output.return_value.invoke = capture

            asyncio.run(
                app.ainvoke(
                    {"messages": [HumanMessage(content="user-b query")]},
                    config_b,
                )
            )

        assert len(captured_states) > 0
        b_messages = captured_states[0]["messages"]
        b_contents = [m.content for m in b_messages]

        assert not any("user-a" in c for c in b_contents), (
            "Thread B should not see Thread A's messages but got: "
            f"{b_contents}"
        )
