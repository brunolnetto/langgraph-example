import os
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from pydantic_ai import Agent


load_dotenv()

# --------------------------
# Estado e tipo do fluxo
# --------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]

class UserIntent(BaseModel):
    intent: Literal["exit", "continue"]

class ConfirmationIntent(str, Enum):
    confirm = "confirm"
    deny = "deny"
    unclear = "unclear"

class ConfirmationOutput(BaseModel):
    intent: ConfirmationIntent = Field(
        ...,
        description="User confirmation intent: 'confirm' if user agreed, 'deny' if refused, or 'unclear' if ambiguous."
    )

# --------------------------
# Ferramentas
# --------------------------

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

search_tool = TavilySearch(max_results=2)
safe_tools = [search_tool]
sensitive_tools = [human_assistance]

# --------------------------
# Models
# --------------------------

LLM_MODEL = os.getenv("LLM_MODEL", "openai:o3-mini")

safe_llm = init_chat_model(LLM_MODEL).bind_tools(safe_tools)
sensitive_llm = init_chat_model(LLM_MODEL).bind_tools(sensitive_tools)

# --------------------------
# Nodes
# --------------------------

def chatbot_with_safe_tools(state: State):
    message = safe_llm.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

def confirm_topic_node(state: State):
    confirm_msg = HumanMessage(content="Do you want me to consult a human on this topic? (yes/no)")
    return {"messages": [confirm_msg]}

def sensitive_tool_handler(state: State):
    message = sensitive_llm.invoke(state["messages"])
    return {"messages": [message]}

def end_conversation_node(state: State):
    message = safe_llm.invoke(state["messages"])
    return {"messages": [message]}

# --------------------------
# Decision agents
# --------------------------

intent_classifier = Agent(
    model=LLM_MODEL,
    output_type=UserIntent,
    system_prompt="""
Classify the user's intent.

Respond only with JSON:

{ "intent": "exit" }      # if user wants to quit
{ "intent": "continue" }  # if user wants to keep chatting
""",
    retries=2,
)

confirmation_classifier = Agent(
    model=LLM_MODEL,
    output_type=ConfirmationOutput,
    system_prompt="""
You are a confirmation intent classifier.

Given a user message, classify it into:
- "confirm" if user agrees
- "deny" if user declines
- "unclear" if ambiguous, sarcastic, or unrelated

Respond only in this JSON format:
{ "intent": "confirm" }
""",
    retries=2,
)

def check_user_intent(state: State) -> str:
    last_user_msg = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    result = intent_classifier.run_sync(last_user_msg.content)
    intent = result.output.intent
    print(f"[Intent → {intent}]")
    return intent

def detect_sensitive_case(state: State) -> bool:
    last_msg = state["messages"][-1]
    return hasattr(last_msg, "tool_calls") and \
        last_msg.tool_calls and \
        last_msg.tool_calls[0]["name"] == "human_assistance_node"

MAX_RETRIES = 3
def handle_user_confirmation(state: State) -> str:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, HumanMessage):
        return "unclear"

    # Initialize retry counter if not present
    retries = state.get("confirmation_retries", 0)

    result = confirmation_classifier.invoke(last_msg.content)
    intent = result.intent.value  # "confirm", "deny", or "unclear"

    if intent == "unclear":
        retries += 1
        state["confirmation_retries"] = retries
        if retries >= MAX_RETRIES:
            print("[Confirmation retries exhausted, defaulting to 'deny']")
            return "deny"
        return "unclear"
    else:
        # Reset retries on clear answer
        if "confirmation_retries" in state:
            del state["confirmation_retries"]
        return intent


# --------------------------
# Graph
# --------------------------

graph_builder = StateGraph(State)

# Node definitions
graph_builder.add_node("which_intent_decision", lambda state: {})
graph_builder.add_node("chatbot_with_safe_tools_node", chatbot_with_safe_tools)
graph_builder.add_node("safe_tools", ToolNode(tools=safe_tools))
graph_builder.add_node("sensitive_tools", ToolNode(tools=sensitive_tools))
graph_builder.add_node("is_sensitive_decision", detect_sensitive_case)
graph_builder.add_node("awaiting_user_confirmation_decision", handle_user_confirmation)
graph_builder.add_node("human_assistance_node", lambda state: {"messages": state.messages})
graph_builder.add_node("end_conversation_node", end_conversation_node)

# Transitions
graph_builder.add_edge(START, "which_intent_decision")

graph_builder.add_conditional_edges(
    "which_intent_decision",
    check_user_intent, 
    {
        "continue": "chatbot_with_safe_tools_node",
        "exit": "end_conversation_node"
    }
)

graph_builder.add_conditional_edges(
    "chatbot_with_safe_tools_node", 
    tools_condition, 
    {
        True: "safe_tools",
        False: "is_sensitive_decision"
    }
)
graph_builder.add_edge("safe_tools", "chatbot_with_safe_tools_node")

graph_builder.add_conditional_edges(
    "is_sensitive_decision", 
    detect_sensitive_case, 
    {
        True: "awaiting_user_confirmation_decision",
        False: "chatbot_with_safe_tools_node"
    }
)

graph_builder.add_conditional_edges(
    "awaiting_user_confirmation_decision", 
    handle_user_confirmation,
    {
        "confirm": "human_assistance_node",
        "deny": "chatbot_with_safe_tools_node",
        "unclear": "awaiting_user_confirmation_decision"
    }
)

graph_builder.add_edge("human_assistance_node", "sensitive_tools")
graph_builder.add_edge("sensitive_tools", "chatbot_with_safe_tools_node")

graph_builder.set_finish_point("end_conversation_node")

# Compilação com checkpoint
graph_obj = graph_builder.compile(checkpointer=MemorySaver())
