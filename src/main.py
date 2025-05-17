from typing import Callable
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from graph import graph_obj
from utils import interactive_human_handler, handle_interrupt, request_user_input

def stream_graph_updates(config, user_input: str) -> str:
    stream_mode = ["values", "debug", "tokens"]
    message = HumanMessage(content=user_input)
    state = {"messages": [message]}
    iterator = graph_obj.stream(state, config, stream_mode=stream_mode)

    human_handler = interactive_human_handler
    terminal_node = None
    last_ai_message = None
    current_node = None

    while True:
        try:
            stream_type, event_obj = next(iterator)
        except StopIteration:
            break

        if stream_type == "debug":
            current_node = event_obj.get("payload", {}).get("name")
            if current_node:
                print(f"📍 Node: {current_node}")

        if "__interrupt__" in event_obj:
            cmd = handle_interrupt(event_obj, human_handler)

            if last_ai_message and last_ai_message.tool_calls:
                first_call = last_ai_message.tool_calls[0]
                if "id" in first_call:
                    tool_response = ToolMessage(
                        tool_call_id=first_call["id"],
                        content=cmd.resume["data"]
                    )
                    state["messages"].append(tool_response)
                    iterator = graph_obj.stream(state, config, stream_mode=stream_mode)
                    continue

            iterator = graph_obj.stream(cmd, config, stream_mode=stream_mode)
            continue

        if "messages" in event_obj:
            last_ai_message = next(
                (msg for msg in reversed(event_obj["messages"]) if msg.type == "ai"),
                None
            )
            if last_ai_message:
                last_ai_message.pretty_print()
                state["messages"].append(last_ai_message)

                # ✅ Extrair token usage
                usage = last_ai_message.response_metadata.get("token_usage", {})
                prompt = usage.get("prompt_tokens", 0)
                completion = usage.get("completion_tokens", 0)
                total = usage.get("total_tokens", 0)
                print(f"🧠 Token usage — Prompt: {prompt}, Completion: {completion}, Total: {total}")

            if current_node == "end_conversation_node" and stream_type == "values":
                terminal_node = "END"
                break

    return terminal_node

# --------------------------------- CLI ---------------------------------
def cli_loop():
    # Hard-coded values: this may change ot become customizable through UI.
    config = {
        "stream_mode": "values",
        "stream": True,
        "stream_interval": 0.1,
        "max_tokens": 100,
        "temperature": 0.5,
        "configurable": {
            "user_id": "default_user",
            "thread_id": "thread-1"
        }
    }

    while True:
        try:
            user_input = request_user_input()
            if not user_input:
                print("⚠️ Empty input. Please type something.")
                continue

            terminal_node = stream_graph_updates(config, user_input)

            if terminal_node == "END":
                print("👋 Goodbye!")
                break

        except (EOFError, KeyboardInterrupt):
            print("\n⚠️ Interrupted. 👋 Goodbye!")
            break

if __name__ == "__main__":
    cli_loop()
