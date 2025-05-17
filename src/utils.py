from typing import Callable

from langgraph.types import Command

from models import State 
def actor_input(actor_emoji: str, actor_name: str):
    return input(f"👤 User (you): ").strip()


def request_user_input():
    return actor_input(f"👤", "User (you)")


def request_human_assistance_input():
    return actor_input("👨🏻‍💻", "Human assistance")


def interactive_human_handler(query: str) -> str:
    print(f"[Graph paused for human assistance: “{query}”]")
    try:
        return request_human_assistance_input()
    except EOFError:
        print("⚠️ Input not available. Using fallback.")
        return "Sorry, no one is available right now."


def handle_interrupt(event, human_handler: Callable[[str], str]) -> Command:
    interrupt_event = event["__interrupt__"][0]
    query = interrupt_event.value.get("query", "[No query provided]")
    human_reply = human_handler(query)
    print(f"[Human → {human_reply}]\n")
    return Command(resume={"data": human_reply})


def tool_handler(llm, state: State):
    message = llm.invoke(state["messages"])
    return {"messages": [message]}