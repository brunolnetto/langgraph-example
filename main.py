import os
import json
from typing import Annotated
from dotenv import load_dotenv

from typing_extensions import TypedDict

from langchain_core.messages import ToolMessage
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

load_dotenv()

class State(TypedDict):
    # Messages have the type "list". The add_messages function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# Available tools are defined here
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


search_tool = TavilySearch(max_results=2)
tools = [search_tool, human_assistance]

LLM_MODEL = os.getenv("LLM_MODEL", "openai:o3-mini")
llm = init_chat_model(LLM_MODEL)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# The tools_condition function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    # Start with the user’s message
    payload = {"messages":[{"role":"user","content":user_input}]}
    events = graph.stream(payload, config, stream_mode="values")

    for evt in events:
        # If we hit an interrupt, mock the human reply
        if "__interrupt__" in set(evt.keys()):
            query = evt["__interrupt__"][0].value["query"]
            print(f"[Graph paused for human assistance: “{query}”]")

            mocked_response = (
                "We, the experts, are here to help! "
                "We’d recommend you check out LangGraph to build your agent."
            )
            print(f"[Mocked Human → {mocked_response}]")

            # Resume via Command
            cmd = Command(resume={"data": mocked_response})
            resumed = graph.stream(cmd, config, stream_mode="values")
            
            # Print out all subsequent messages
            for ev2 in resumed:
                if "messages" in ev2:
                    ev2["messages"][-1].pretty_print()
            return  # done after handling one interrupt

        # Otherwise just print LLM/tool messages
        if "messages" in evt:
            evt["messages"][-1].pretty_print()

quit_commands = ["exit", "quit", "q", "bye", "goodbye"]

while True:
    try:
        user_input = input(f"User [type ({", ".join(quit_commands)})  to quit]: ")
        if user_input.lower() in quit_commands:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break