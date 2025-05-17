from typing_extensions import TypedDict
from typing import Annotated, Literal
from enum import Enum

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

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