"""Define the configurable parameters for the Neo4j chatbot agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Annotated


@dataclass(kw_only=True)
class Context:
    """The context for the Neo4j chatbot agent."""

    system_prompt: str = field(
        default="""You are an expert assistant for analyzing a Science Diplomacy conference knowledge graph stored in Neo4j.

Your goal is to provide accurate, insightful answers by intelligently selecting and using the appropriate search tools:

- **semantic_search**: For content-focused queries about themes, concepts, and meanings in Answer, Question, or Talkingpoint nodes
- **graph_search**: For structural queries about relationships, attributions, speakers, organizations, sessions, and connections  
- **hybrid_search**: For complex queries requiring both content analysis and relationship traversal

Always ground your responses in the actual data from the knowledge graph. If no relevant information is found, say so explicitly.""",
        metadata={
            "description": "The system prompt for the Neo4j chatbot agent"
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4",
        metadata={
            "description": "The language model to use for the agent"
        },
    )

    neo4j_uri: str = field(
        default="",
        metadata={
            "description": "Neo4j database URI"
        },
    )

    neo4j_username: str = field(
        default="",
        metadata={
            "description": "Neo4j username"
        },
    )

    neo4j_password: str = field(
        default="",
        metadata={
            "description": "Neo4j password"  
        },
    )

    neo4j_database: str = field(
        default="neo4j",
        metadata={
            "description": "Neo4j database name"
        },
    )

    openai_api_key: str = field(
        default="",
        metadata={
            "description": "OpenAI API key"
        },
    )

    def __post_init__(self) -> None:
        """Load environment variables if not provided."""
        env_mapping = {
            "neo4j_uri": "NEO4J_URI",
            "neo4j_username": "NEO4J_USERNAME", 
            "neo4j_password": "NEO4J_PASSWORD",
            "neo4j_database": "NEO4J_DATABASE",
            "openai_api_key": "OPENAI_API_KEY"
        }
        
        for attr, env_var in env_mapping.items():
            if not getattr(self, attr):
                setattr(self, attr, os.environ.get(env_var, ""))