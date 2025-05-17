import os
from dotenv import load_dotenv
from typing import Annotated, Literal


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from pydantic_ai import Agent

from models import State, UserIntent, ConfirmationOutput
from utils import tool_handler

load_dotenv()

# --------------------------
# Ferramentas
# --------------------------
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tools = [TavilySearch(max_results=2), human_assistance]

# --------------------------
# Models
# --------------------------

LLM_PROVIDER = 'openai'
LLM_MODEL = "o3-mini"

llm = init_chat_model(
    model_provider=LLM_PROVIDER, model=LLM_MODEL
).bind_tools(tools)

# --------------------------
# Decision agents
# --------------------------

intent_prompt="""
Classify the user's intent.

Respond only with JSON:

{ "intent": "exit" }      # if user wants to quit
{ "intent": "continue" }  # if user wants to keep chatting
"""
intent_classifier = Agent(
    model=LLM_MODEL,
    output_type=UserIntent,
    system_prompt=intent_prompt,
    retries=2,
)

# --------------------------
# Decision conditions
# --------------------------

def check_user_intent(state: State) -> str:
    last_user_msg = next(
        m for m in reversed(state["messages"]) 
        if isinstance(m, HumanMessage)
    )
    result = intent_classifier.run_sync(last_user_msg.content)
    intent = result.output.intent
    print(f"[Intent → {intent}]")
    return intent

# --------------------------
# Nodes
# --------------------------

def chatbot(state: State):
    message = llm.invoke(state["messages"])
    return {"messages": [message]}

def end_conversation_node(state: State):
    return chatbot(state)

def chabot_tool_handler(state: State):
    return tool_handler(llm, state)

# --------------------------
# Graph
# --------------------------

graph_builder = StateGraph(State)

# Node definitions
graph_builder.add_node("intent_decision", lambda state: {})
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("end_conversation_node", end_conversation_node)

# Unconditional transitions
graph_builder.add_edge(START, "intent_decision")
graph_builder.add_edge("tools", "chatbot")

# Condition transitions
graph_builder.add_conditional_edges(
    "intent_decision", check_user_intent, 
    {
        "continue": "chatbot",
        "exit": "end_conversation_node"
    }
)

graph_builder.add_conditional_edges("chatbot", tools_condition)

# Overwrites finished point
graph_builder.set_finish_point("end_conversation_node")

# Compilação com checkpoint
graph_obj = graph_builder.compile(checkpointer=MemorySaver())
