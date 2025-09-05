"""Search tools for the Neo4j chatbot agent."""

from typing import Optional
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing_extensions import Literal
from llm_utils import make_chat_model

# Valid node types for semantic search
SEMANTIC_NODE_TYPES = ['Question', 'Answer', 'Talkingpoint']

# Index mapping for each semantic node type
INDEX_BY_LABEL = {
    "Answer": "answer_embeddings",
    "Question": "question_embeddings", 
    "Talkingpoint": "talkingpoint_embeddings",
}

# Global cache for vector indexes and graph connections
vector_index_cache = {}
graph_cache = None
_last_graph_chain_error: Optional[str] = None


def get_vector_index(node_label: str, index_name: str) -> Optional[Neo4jVector]:
    """Get existing vector index for Question, Answer, or Talkingpoint nodes."""
    if node_label not in SEMANTIC_NODE_TYPES:
        return None
        
    cache_key = f"{node_label}_{index_name}"
    if cache_key in vector_index_cache:
        return vector_index_cache[cache_key]
        
    try:
        vector_index = Neo4jVector.from_existing_index(
            OpenAIEmbeddings(),
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            index_name=index_name,
        )
        vector_index_cache[cache_key] = vector_index
        return vector_index
    except Exception:
        return None


def get_graph_chain() -> Optional[GraphCypherQAChain]:
    """Get or create Neo4j graph chain."""
    global graph_cache, _last_graph_chain_error
    
    if graph_cache is not None:
        return graph_cache
    
    try:
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )
        graph.refresh_schema()
        
        cypher_model = os.getenv("CYPHER_LLM_MODEL", "openai/gpt-4")
        qa_model = os.getenv("QA_LLM_MODEL", cypher_model)
        cypher_temp_str = os.getenv("CYPHER_TEMPERATURE") or os.getenv("DEFAULT_TEMPERATURE", "")
        qa_temp_str = os.getenv("QA_TEMPERATURE") or os.getenv("DEFAULT_TEMPERATURE", "")
        cypher_temp = None
        qa_temp = None
        if cypher_temp_str and cypher_temp_str.strip().lower() not in ("none", "null"):
            try:
                cypher_temp = float(cypher_temp_str)
            except ValueError:
                cypher_temp = None
        if qa_temp_str and qa_temp_str.strip().lower() not in ("none", "null"):
            try:
                qa_temp = float(qa_temp_str)
            except ValueError:
                qa_temp = None

        cypher_kwargs = {"max_tokens": 4096}
        if cypher_temp is not None:
            cypher_kwargs["temperature"] = cypher_temp
        qa_kwargs = {"max_tokens": 4096}
        if qa_temp is not None:
            qa_kwargs["temperature"] = qa_temp

        graph_cache = GraphCypherQAChain.from_llm(
            cypher_llm=make_chat_model(cypher_model, **cypher_kwargs),
            qa_llm=make_chat_model(qa_model, **qa_kwargs),
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True
        )
        _last_graph_chain_error = None
        return graph_cache
    except Exception as e:
        _last_graph_chain_error = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"[graph_chain] initialization error: {_last_graph_chain_error}")
        return None


class SemanticSearchArgs(BaseModel):
    query: str = Field(..., description="The natural language query to search for.")
    node_label: Literal["Answer", "Question", "Talkingpoint"] = Field(
        "Answer", description="The node label to search over for semantic meaning."
    )
    reason: str = Field(
        ..., description="Single-sentence justification for using this tool for this step (no chain-of-thought)."
    )


@tool("semantic_search", args_schema=SemanticSearchArgs)
async def semantic_search(query: str, node_label: str = "Answer", reason: str = "") -> str:
    """Content-only semantic retrieval over Answer|Question|Talkingpoint text.
    Use for themes and meaning in the text; not for attribution, counts, or relationship traversal.
    Args:
    - query: user query text
    - node_label: one of Answer | Question | Talkingpoint
    - reason: REQUIRED. Single-sentence justification for calling semantic_search for this step (no chain-of-thought)."""
    
    # Handle JSON string input from agent
    import json
    try:
        if query.strip().startswith('{') and 'node_label' in query:
            params = json.loads(query)
            actual_query = params.get('query', query)
            actual_node_label = params.get('node_label', node_label)
            return await _semantic_search_impl(actual_query, actual_node_label)
    except (json.JSONDecodeError, KeyError):
        pass
        
    return await _semantic_search_impl(query, node_label)


async def _semantic_search_impl(query: str, node_label: str) -> str:
    """Internal implementation for semantic search."""
    if node_label not in SEMANTIC_NODE_TYPES:
        return f"node_label must be one of {SEMANTIC_NODE_TYPES}"
        
    index_name = INDEX_BY_LABEL.get(node_label, "embeddings")
    print(f"🔍 Searching {node_label} nodes for: '{query}' (index: {index_name})")
    
    vector_index = get_vector_index(node_label, index_name)
    if not vector_index:
        return f"No vector index available for {node_label}. Expected index: {index_name}."
    
    try:
        qa_model = os.getenv("QA_LLM_MODEL", os.getenv("CYPHER_LLM_MODEL", "openai/gpt-4"))
        qa_temp_str = os.getenv("QA_TEMPERATURE") or os.getenv("DEFAULT_TEMPERATURE", "")
        qa_temp = None
        if qa_temp_str and qa_temp_str.strip().lower() not in ("none", "null"):
            try:
                qa_temp = float(qa_temp_str)
            except ValueError:
                qa_temp = None
        qa_kwargs = {"max_tokens": 4096}
        if qa_temp is not None:
            qa_kwargs["temperature"] = qa_temp
        vector_qa = RetrievalQA.from_chain_type(
            llm=make_chat_model(qa_model, **qa_kwargs),
            chain_type="stuff",
            retriever=vector_index.as_retriever(),
        )
        result = await vector_qa.ainvoke({"query": query})
        return result.get("result", str(result)) if isinstance(result, dict) else str(result)
    except Exception as e:
        return f"Semantic search failed for {node_label}: {str(e)[:100]}..."


class GraphSearchArgs(BaseModel):
    query: str = Field(..., description="The structural/attribution query to answer with Cypher.")
    reason: str = Field(..., description="Single-sentence justification for using this tool for this step (no chain-of-thought).")


@tool("graph_search", args_schema=GraphSearchArgs)
async def graph_search(query: str, reason: str) -> str:
    """Cypher-based graph traversal for structural/attribution queries (who/what/where).
    Use for structural queries like affiliations, sessions, references, topic connections, counts, traversals, and talkingpoint relationships.
    Not for content-only semantics—use semantic_search for meaning.
    Args:
    - query: natural-language structural question to answer with Cypher
    - reason: REQUIRED. Single-sentence justification for calling graph_search for this step (no chain-of-thought)."""
    cypher_chain = get_graph_chain()
    if not cypher_chain:
        # Bubble up the last recorded initialization error for clarity
        detail = _last_graph_chain_error or "Could not connect to Neo4j"
        return f"Graph search unavailable: {detail}"
    
    try:
        result = await cypher_chain.ainvoke({"query": query})
        return result.get("result", str(result)) if isinstance(result, dict) else str(result)
    except Exception as e:
        return f"Graph search failed: {str(e)[:100]}..."


class HybridSearchArgs(BaseModel):
    query: str = Field(
        ..., description="The query requiring both content and relationship reasoning."
    )
    node_label: Literal["Answer", "Question", "Talkingpoint"] = Field(
        "Answer", description="The node label to use for the semantic component."
    )
    reason: str = Field(..., description="Single-sentence justification for using this tool for this step (no chain-of-thought).")


@tool("hybrid_search", args_schema=HybridSearchArgs)
async def hybrid_search(query: str, node_label: str = "Answer", reason: str = "") -> str:
    """Combines semantic_search (content) and graph_search (structure) to answer mixed queries.
    Provide node_label for the semantic component (Answer | Question | Talkingpoint).
    Use when the query mixes content meaning with attribution/relationships (e.g., 'who said what about X').
    Args:
    - query: user query text
    - node_label: semantic target label
    - reason: REQUIRED. Single-sentence justification for choosing hybrid_search for this step (no chain-of-thought)."""
    
    # Handle JSON string input from agent
    import json
    actual_query = query
    actual_node_label = node_label
    try:
        if query.strip().startswith('{') and 'node_label' in query:
            params = json.loads(query)
            actual_query = params.get('query', query)
            actual_node_label = params.get('node_label', node_label)
    except (json.JSONDecodeError, KeyError):
        pass
        
    print(f"🔍 Executing hybrid search for: {actual_query}")
    
    # Use parameter-driven approach
    index_name = INDEX_BY_LABEL.get(actual_node_label, "embeddings")
    
    # Run semantic search
    print(f"📊 Running semantic search on {actual_node_label} nodes...")
    try:
        semantic_results = await _semantic_search_impl(actual_query, actual_node_label)
    except Exception as e:
        semantic_results = f"Semantic search unavailable: {str(e)[:100]}..."
    
    # Run graph search  
    print("🔗 Running graph structure search...")
    try:
        # Invoke the structured tool asynchronously to avoid sync invocation errors
        graph_results = await graph_search.ainvoke({"query": actual_query, "reason": f"From hybrid_search: structural relationships for '{actual_query}'"})
    except Exception as e:
        graph_results = f"Graph search unavailable: {str(e)[:100]}..."
    
    return f"""Hybrid analysis for: "{actual_query}"

📊 CONTENT INSIGHTS ({actual_node_label} nodes):
{semantic_results}

🔗 STRUCTURAL RELATIONSHIPS:
{graph_results}

✅ SYNTHESIS: Combining conference content with relationship patterns to provide comprehensive understanding of the science diplomacy discussion."""


# List of all available tools
TOOLS = [semantic_search, graph_search, hybrid_search]