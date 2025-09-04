"""Define the state structures for the Neo4j chatbot agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Defines the input state for the Neo4j chatbot agent."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


@dataclass
class State(InputState):
    """Represents the complete state of the Neo4j chatbot agent."""

    is_last_step: IsLastStep = field(default=False)