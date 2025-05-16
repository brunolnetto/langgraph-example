from typing import Callable
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from graph import graph_obj

# ----------- Config -----------

config = {"configurable": {"thread_id": "1"}}

def interactive_human_handler(query: str) -> str:
    print(f"[Graph paused for human assistance: “{query}”]")
    try:
        return input("👨🏻‍💻 Human assistance: ").strip()
    except EOFError:
        print("⚠️ Input not available. Using fallback.")
        return "Sorry, no one is available right now."


def handle_interrupt(event, human_handler: Callable[[str], str]) -> Command:
    interrupt_event = event["__interrupt__"][0]
    query = interrupt_event.value.get("query", "[No query provided]")
    human_reply = human_handler(query)
    print(f"[Human → {human_reply}]\n")
    return Command(resume={"data": human_reply})

# ----------- Core Loop -----------

def stream_graph_updates(user_input: str) -> str:
    stream_mode = ["values", "debug"]
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

        # Debug: track current node
        if stream_type == "debug":
            current_node = event_obj.get("payload", {}).get("name")
            if current_node:
                print(f"📍 Node: {current_node}")

        # Handle interrupt (tool-based human input)
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

        # Capture assistant message
        if "messages" in event_obj:
            last_ai_message = next(
                (msg for msg in reversed(event_obj["messages"]) if msg.type == "ai"),
                None
            )
            if last_ai_message:
                last_ai_message.pretty_print()
                state["messages"].append(last_ai_message)

                # If we're in `confirm_topic`, ask the user to confirm
                if current_node == "awaiting_user_confirmation":
                    user_confirmation = input("🧠 Do you want to consult a human? (yes/no): ").strip().lower()
                    confirmation_msg = HumanMessage(content=user_confirmation)
                    state["messages"].append(confirmation_msg)
                    iterator = graph_obj.stream(state, config, stream_mode=stream_mode)
                    continue

        if current_node == "end_conversation" and stream_type == "values":
            terminal_node = "END"
            break

    return terminal_node

# ----------- CLI -----------

def cli_loop():
    while True:
        try:
            user_input = input("👤 User (you): ").strip()
            if not user_input:
                print("⚠️ Empty input. Please type something.")
                continue

            terminal_node = stream_graph_updates(user_input)
            if terminal_node == "END":
                print("👋 Goodbye!")
                break

        except (EOFError, KeyboardInterrupt):
            print("\n⚠️ Interrupted. 👋 Goodbye!")
            break

if __name__ == "__main__":
    cli_loop()
