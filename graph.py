"""Define the LangGraph ReAct agent for Neo4j chatbot."""

from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from llm_utils import make_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from state import InputState, State
from tools import TOOLS


async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering the Neo4j chatbot agent."""
    
    # Get configuration from config
    configurable = config.get("configurable", {})
    model_name = configurable.get("model", "gpt-4")
    system_prompt = configurable.get("system_prompt", "You are a helpful assistant for analyzing a Science Diplomacy conference knowledge graph.")
    temperature = configurable.get("temperature", None)
    
    # Initialize the model with tool binding and proper token limits
    # Set max_tokens based on model capabilities
    if "gpt-4" in model_name:
        max_tokens = 4096
    elif "gpt-3.5" in model_name:
        max_tokens = 4096
    else:
        max_tokens = 4096  # Default for other models
        
    # Build kwargs allowing optional temperature
    model_kwargs = {"max_tokens": max_tokens}
    if temperature is not None:
        model_kwargs["temperature"] = float(temperature)

    model = make_chat_model(
        model_name, **model_kwargs
    ).bind_tools(TOOLS)

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_prompt}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response
    return {"messages": [response]}


# Define the graph
builder = StateGraph(State)

# Define nodes
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set entrypoint
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add conditional edge
builder.add_conditional_edges(
    "call_model",
    route_model_output,
)

# Add edge from tools back to model
builder.add_edge("tools", "call_model")

# Compile the graph
graph = builder.compile()