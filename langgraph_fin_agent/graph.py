import os
import functools
import operator
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from .tools import (
    get_stock_price,
    get_company_profile,
    get_financial_ratios,
    get_key_metrics,
    get_stock_screener,
    generate_single_line_item_query,
    read_webpage,
)

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# Cache repeated LLM calls (e.g. supervisor routing on the same conversation state).
set_llm_cache(InMemoryCache())

MEMBERS = ["Financial_Data_Agent", "Web_Research_Agent", "Output_Summarizing_Agent"]
OPTIONS = ("FINISH",) + tuple(MEMBERS)

FINANCIAL_DATA_TOOLS = [
    get_stock_price,
    get_company_profile,
    get_financial_ratios,
    get_key_metrics,
    get_stock_screener,
    generate_single_line_item_query,
]

WEB_RESEARCH_TOOLS = [
    read_webpage,
]

LLM = ChatOpenAI(model="gpt-4o-mini")

ORCHESTRATOR_SYSTEM_PROMPT = """
You are a supervisor routing tasks between agents: {members}.

- Financial_Data_Agent: fetches stock prices, company profiles, financial ratios, key metrics, and screener data via APIs.
- Web_Research_Agent: scrapes content from URLs when additional context from the web is needed.
- Output_Summarizing_Agent: synthesizes all gathered data into a final response for the user.

Route to the appropriate agent based on what the task requires. Once sufficient data has been collected, route to Output_Summarizing_Agent, then respond with FINISH.
"""

FINANCIAL_DATA_SYSTEM_PROMPT = """
You are a financial data agent responsible for retrieving financial data using the provided API tools.

Your tasks:
1. Given a user query, use the appropriate tool to fetch the data you believe relevant to answer the query.
2. Return only the raw data obtained from the tool, instead of trying to answer the query as that's handled by other agents.
3. Do not add commentary, explanations, or infer information beyond the tool's output.
4. Remember that data interpretation, calculation, and analysis is handled by other agents.

Always provide the unprocessed data as your response.
"""

OUTPUT_SUMMARIZING_SYSTEM_PROMPT = """
You are an output summarizing agent responsible for synthesizing information from other agents.

Your tasks:
1. Analyze data and calculations from the Financial Data and Calculator Agents.
2. Provide a clear, concise summary of key findings and results.
3. Ensure the summary directly addresses the user's original query.
4. Use table format for data presentation when appropriate to improve readability.

Prioritize clarity, relevance, and user-friendly presentation in your summaries.
"""

WEB_RESEARCH_SYSTEM_PROMPT = """
You are a web research agent responsible for gathering information from websites.

Your tasks:
1. Given a user query, use the web scraping tool to gather relevant information from provided URLs.
2. Focus on extracting key information that helps answer the user's query.
3. Return the relevant extracted information without additional commentary.
4. Remember that data interpretation and analysis is handled by other agents.

Always provide the extracted information as your response.
"""


class RouteResponse(BaseModel):
    next: Literal[OPTIONS]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


async def agent_node(state, agent, name):
    try:
        result = await agent.ainvoke(state)

        if isinstance(result, dict) and "messages" in result:
            return {
                "messages": [
                    AIMessage(content=result["messages"][-1].content, name=name)
                ]
            }
        return {"messages": [AIMessage(content=str(result), name=name)]}
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"An error occurred: {str(e)}", name=name)]
        }


def supervisor_agent(state):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ORCHESTRATOR_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(OPTIONS), members=", ".join(MEMBERS))

    # Pass only the last 4 messages so the routing prompt doesn't balloon as
    # tool results accumulate in the shared state across steps.
    trimmed_state = {**state, "messages": state["messages"][-4:]}
    supervisor_chain = prompt | LLM.with_structured_output(RouteResponse)
    return supervisor_chain.invoke(trimmed_state)


def output_summarizing_node(state):
    """Summarize collected agent results into a final user-facing response.

    Only agent/tool messages from the most recent human turn are forwarded
    so the summarizer doesn't re-ingest the full accumulated history
    (including raw API payloads from earlier steps).
    """
    all_messages = state["messages"]

    # Anchor to the last human query.
    last_human_idx = 0
    for i, msg in enumerate(all_messages):
        if hasattr(msg, "type") and msg.type == "human":
            last_human_idx = i

    relevant = all_messages[last_human_idx:]

    # Collect only agent summaries and tool results (capped per message to
    # avoid re-inflating context with raw API responses).
    data_parts = []
    for msg in relevant:
        if hasattr(msg, "type") and msg.type == "tool":
            data_parts.append(f"[Tool result] {str(msg.content)[:2000]}")
        elif hasattr(msg, "name") and msg.name in ("Financial_Data_Agent", "Web_Research_Agent"):
            data_parts.append(f"[{msg.name}] {msg.content}")

    data_block = "\n\n".join(data_parts) if data_parts else "\n".join(
        msg.content for msg in relevant
    )

    messages = [
        ("system", OUTPUT_SUMMARIZING_SYSTEM_PROMPT),
        ("human", f"Please summarize the following information:\n\n{data_block}"),
    ]
    response = LLM.invoke(messages)
    return {
        "messages": [
            AIMessage(content=response.content, name="Output_Summarizing_Agent")
        ]
    }


financial_data_agent = create_react_agent(
    LLM, tools=FINANCIAL_DATA_TOOLS, state_modifier=FINANCIAL_DATA_SYSTEM_PROMPT
)

financial_data_node = functools.partial(
    agent_node,  # Reusable function to convert an agent into a node
    agent=financial_data_agent,
    name="Financial_Data_Agent",
)

web_research_agent = create_react_agent(
    LLM, tools=WEB_RESEARCH_TOOLS, state_modifier=WEB_RESEARCH_SYSTEM_PROMPT
)

web_research_node = functools.partial(
    agent_node, agent=web_research_agent, name="Web_Research_Agent"
)

from langgraph.graph import END, START, StateGraph

def build_workflow() -> StateGraph:
    """Construct the state graph for the workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("Supervisor_Agent", supervisor_agent)
    workflow.add_node("Financial_Data_Agent", financial_data_node)
    workflow.add_node("Web_Research_Agent", web_research_node)
    workflow.add_node("Output_Summarizing_Agent", output_summarizing_node)

    workflow.add_edge("Financial_Data_Agent", "Supervisor_Agent")
    workflow.add_edge("Web_Research_Agent", "Supervisor_Agent")

    conditional_map = {
        "Financial_Data_Agent": "Financial_Data_Agent",
        "Web_Research_Agent": "Web_Research_Agent",
        "Output_Summarizing_Agent": "Output_Summarizing_Agent",
        "FINISH": "Output_Summarizing_Agent",
    }
    workflow.add_conditional_edges(
        "Supervisor_Agent", lambda x: x["next"], conditional_map
    )

    workflow.add_edge("Output_Summarizing_Agent", END)
    workflow.add_edge(START, "Supervisor_Agent")

    return workflow


def build_app():
    """Build and compile the workflow."""
    memory = MemorySaver()
    workflow = build_workflow()

    return workflow.compile(checkpointer=memory)
