import os
import functools
import operator
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    get_market_cap,
    get_stock_screener,
    generate_single_line_item_query,
    read_webpage,
)
from .prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    FINANCIAL_DATA_SYSTEM_PROMPT,
    OUTPUT_SUMMARIZING_SYSTEM_PROMPT,
    WEB_RESEARCH_SYSTEM_PROMPT,
)

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

MEMBERS = ["Financial_Data_Agent", "Web_Research_Agent", "Output_Summarizing_Agent"]
OPTIONS = ("FINISH",) + tuple(MEMBERS)

FINANCIAL_DATA_TOOLS = [
    get_stock_price,
    get_company_profile,
    get_financial_ratios,
    get_key_metrics,
    get_market_cap,
    get_stock_screener,
    generate_single_line_item_query,
]

WEB_RESEARCH_TOOLS = [
    read_webpage,
]

LLM = ChatOpenAI(model="gpt-4o-mini")


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

    supervisor_chain = prompt | LLM.with_structured_output(RouteResponse)
    return supervisor_chain.invoke(state)


def output_summarizing_node(state):
    """Process the state and generate a summary using the LLM."""
    messages = [
        ("system", OUTPUT_SUMMARIZING_SYSTEM_PROMPT),
        (
            "assistant",
            "Please summarize the following information:\n\n"
            + "\n".join([msg.content for msg in state["messages"]]),
        ),
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
