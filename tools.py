"""Search tools for the Neo4j chatbot agent."""

from typing import Optional
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing_extensions import Literal
from llm_utils import make_chat_model
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# Valid node types for semantic search
SEMANTIC_NODE_TYPES = ['Question', 'Answer', 'Talkingpoint']

# Index mapping for each semantic node type
INDEX_BY_LABEL = {
    "Answer": "answer_embeddings",
    "Question": "question_embeddings", 
    "Talkingpoint": "talkingpoint_embeddings",
}

# Search configuration with environment defaults
def get_search_config():
    """Get search configuration from environment variables with defaults."""
    try:
        k = int(os.getenv('SEMANTIC_SEARCH_K', '5'))
        k = max(1, min(k, 50))  # Clamp between 1 and 50
        
        similarity_threshold = float(os.getenv('SEMANTIC_SEARCH_SIMILARITY_THRESHOLD', '0.0'))
        similarity_threshold = max(0.0, min(similarity_threshold, 1.0))  # Clamp between 0.0 and 1.0
        
        reranker_top_k = int(os.getenv('SEMANTIC_SEARCH_RERANKER_TOP_K', '3'))
        reranker_top_k = max(1, min(reranker_top_k, k))  # Clamp between 1 and k
        
        return {
            'k': k,
            'similarity_threshold': similarity_threshold,
            'use_reranker': os.getenv('SEMANTIC_SEARCH_USE_RERANKER', 'false').lower() == 'true',
            'reranker_type': os.getenv('SEMANTIC_SEARCH_RERANKER_TYPE', 'none'),
            'reranker_top_k': reranker_top_k
        }
    except (ValueError, TypeError) as e:
        print(f"Warning: Invalid search configuration, using defaults: {e}")
        return {
            'k': 5,
            'similarity_threshold': 0.0,
            'use_reranker': False,
            'reranker_type': 'none',
            'reranker_top_k': 3
        }

def _apply_reranker(retriever, config):
    """Apply reranker to retriever based on configuration using native LangChain parameters."""
    try:
        reranker_type = config['reranker_type'].lower()
        reranker_top_k = config['reranker_top_k']
        similarity_threshold = config['similarity_threshold']
        
        if reranker_type == 'cohere':
            # Try to use Cohere reranker with native parameters
            try:
                print(f"[DEBUG] Attempting to import CohereRerank...")
                from langchain.retrievers import ContextualCompressionRetriever
                print(f"[DEBUG] ContextualCompressionRetriever imported successfully")
                from langchain_cohere import CohereRerank
                print(f"[DEBUG] CohereRerank imported successfully")
                
                cohere_api_key = os.getenv('COHERE_API_KEY')
                if not cohere_api_key:
                    print("Warning: COHERE_API_KEY not found, skipping reranker")
                    return retriever
                
                # Use CohereRerank with post-rerank similarity filtering
                cohere_model = os.getenv('COHERE_RERANK_MODEL', 'rerank-english-v3.0')
                compressor_kwargs = {
                    'model': cohere_model,
                    'cohere_api_key': cohere_api_key,
                    'top_n': reranker_top_k
                }
                
                compressor = CohereRerank(**compressor_kwargs)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=retriever
                )
                
                print(f"[DEBUG] CohereRerank applied successfully (model: {cohere_model}, top_n: {reranker_top_k})")
                return compression_retriever
            except ImportError:
                print("Warning: CohereRerank not available, skipping reranker")
                return retriever
                
        
        # If reranker type is unknown or 'none', return original retriever
        return retriever
        
    except Exception as e:
        print(f"Warning: Failed to apply reranker: {str(e)[:100]}...")
        return retriever


def _apply_post_rerank_filtering(documents, similarity_threshold):
    """Apply similarity threshold filtering to documents after reranking."""
    if similarity_threshold <= 0.0:
        return documents
    
    filtered_docs = []
    for doc in documents:
        # CohereRerank stores relevance score in metadata
        relevance_score = doc.metadata.get('relevance_score', 1.0)  # Default to 1.0 if no score
        if relevance_score >= similarity_threshold:
            filtered_docs.append(doc)
        else:
            print(f"[DEBUG] Filtering out document with relevance_score {relevance_score} < {similarity_threshold}")
    
    print(f"[DEBUG] Post-rerank filtered {len(documents)} -> {len(filtered_docs)} documents with threshold {similarity_threshold}")
    return filtered_docs


# Global cache for vector indexes and graph connections
vector_index_cache = {}
graph_cache = None
_last_graph_chain_error: Optional[str] = None


def _read_text_file(path: str) -> Optional[str]:
    """Read a text file and return its contents, or None on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read prompt file '{path}': {str(e)[:120]}...")
        return None


def _get_prompt_from_env(text_env: str, file_env: str) -> Optional[str]:
    """Return prompt text from an env var or a file env var if present."""
    text = os.getenv(text_env)
    if text and text.strip():
        return text
    file_path = os.getenv(file_env)
    if file_path and file_path.strip():
        return _read_text_file(file_path.strip())
    return None


def _build_prompt_template(text_env: str, file_env: str, input_variables: list[str], label: str) -> Optional[PromptTemplate]:
    """Build a PromptTemplate from env/file if provided; else None."""
    prompt_text = _get_prompt_from_env(text_env, file_env)
    if prompt_text and prompt_text.strip():
        print(f"[DEBUG] Using custom {label} prompt from {'env' if os.getenv(text_env) else 'file'}")
        try:
            return PromptTemplate(input_variables=input_variables, template=prompt_text)
        except Exception as e:
            print(f"Warning: Failed to build {label} PromptTemplate: {str(e)[:120]}...")
            return None
    return None


def get_vector_index(node_label: str, index_name: str) -> Optional[Neo4jVector]:
    """Get existing vector index for Question, Answer, or Talkingpoint nodes."""
    if node_label not in SEMANTIC_NODE_TYPES:
        return None
        
    cache_key = f"{node_label}_{index_name}"
    if cache_key in vector_index_cache:
        return vector_index_cache[cache_key]
        
    try:
        # Ensure the retriever returns the correct content property for each label.
        # Talkingpoint should surface `resolved_text` to the LLM, not `text`.
        text_prop = "resolved_text" if node_label == "Talkingpoint" else "text"
        vector_index = Neo4jVector.from_existing_index(
            OpenAIEmbeddings(),
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            index_name=index_name,
            text_node_property=text_prop,
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

        # Build optional custom prompts for GraphCypherQAChain
        cypher_prompt = _build_prompt_template(
            text_env="CYPHER_GENERATION_PROMPT",
            file_env="CYPHER_GENERATION_PROMPT_FILE",
            input_variables=["schema", "question"],
            label="cypher generation",
        )
        graph_qa_prompt = _build_prompt_template(
            text_env="GRAPH_QA_PROMPT",
            file_env="GRAPH_QA_PROMPT_FILE",
            input_variables=["context", "question"],
            label="graph QA",
        )
        extra_kwargs = {}
        if cypher_prompt is not None:
            extra_kwargs["cypher_prompt"] = cypher_prompt
        if graph_qa_prompt is not None:
            extra_kwargs["qa_prompt"] = graph_qa_prompt

        graph_cache = GraphCypherQAChain.from_llm(
            cypher_llm=make_chat_model(cypher_model, **cypher_kwargs),
            qa_llm=make_chat_model(qa_model, **qa_kwargs),
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
            **extra_kwargs,
        )
        _last_graph_chain_error = None
        return graph_cache
    except Exception as e:
        _last_graph_chain_error = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"[graph_chain] initialization error: {_last_graph_chain_error}")
        return None


class SemanticSearchArgs(BaseModel):
    query: str = Field(..., description="A single, specific query made by the user")
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
    - query: A single, specific query made by the user
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
    """Internal implementation for semantic search with configurable parameters."""
    if node_label not in SEMANTIC_NODE_TYPES:
        return f"node_label must be one of {SEMANTIC_NODE_TYPES}"
        
    # Get search configuration from environment
    config = get_search_config()
    
    index_name = INDEX_BY_LABEL.get(node_label, "embeddings")
    print(f"🔍 Searching {node_label} nodes for: '{query}' (index: {index_name}, k={config['k']}, threshold={config['similarity_threshold']}, reranker={config['use_reranker']})")
    
    vector_index = get_vector_index(node_label, index_name)
    if not vector_index:
        return f"No vector index available for {node_label}. Expected index: {index_name}."
    
    try:
        # Configure initial retriever to get more documents for reranking
        initial_k = config['k'] if not config['use_reranker'] else max(config['k'] * 2, config['reranker_top_k'] * 3)
        search_kwargs = {"k": initial_k}
        
        retriever = vector_index.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        
        # Check if we're using reranker for post-filtering approach
        use_reranker_with_filtering = config['use_reranker'] and config['reranker_type'] != 'none'
        
        # Optional custom semantic QA prompt (input variables: context, question)
        custom_semantic_prompt = _build_prompt_template(
            text_env="SEMANTIC_QA_PROMPT",
            file_env="SEMANTIC_QA_PROMPT_FILE",
            input_variables=["context", "question"],
            label="semantic QA",
        )
        
        if use_reranker_with_filtering:
            # Apply reranker and handle post-rerank filtering manually
            retriever = _apply_reranker(retriever, config)
            
            # Get documents from reranker, then apply post-filtering
            docs = retriever.invoke(query)
            
            # Apply similarity threshold filtering to reranked results
            if config['similarity_threshold'] > 0:
                docs = _apply_post_rerank_filtering(docs, config['similarity_threshold'])
            
            # Create QA model
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
            
            # Use documents directly with modern stuff chain
            if custom_semantic_prompt is not None:
                prompt = ChatPromptTemplate.from_template(custom_semantic_prompt.template)
            else:
                prompt = ChatPromptTemplate.from_template(
                    "Use the context to answer.\n\n{context}\n\nQuestion: {question}"
                )
            llm = make_chat_model(qa_model, **qa_kwargs)
            qa_chain = create_stuff_documents_chain(llm, prompt)
            
            # Run QA on filtered documents
            result = await qa_chain.ainvoke({
                "context": docs,
                "question": query
            })
            
        else:
            # Original approach for non-reranker cases
            if config['similarity_threshold'] > 0:
                # Apply manual similarity filtering if no reranker is used
                search_kwargs["score_threshold"] = config['similarity_threshold']
                retriever = vector_index.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs=search_kwargs
                )
            
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
                
            chain_type_kwargs = {"prompt": custom_semantic_prompt} if custom_semantic_prompt is not None else {}
            vector_qa = RetrievalQA.from_chain_type(
                llm=make_chat_model(qa_model, **qa_kwargs),
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs or None,
            )
            result = await vector_qa.ainvoke({"query": query})
        return result.get("result", str(result)) if isinstance(result, dict) else str(result)
    except Exception as e:
        return f"Semantic search failed for {node_label}: {str(e)[:100]}..."


class GraphSearchArgs(BaseModel):
    query: str = Field(..., description="Natural-language structural question to answer with Cypher.")
    reason: str = Field(..., description="Single-sentence justification for using this tool for this step (no chain-of-thought).")


@tool("graph_search", args_schema=GraphSearchArgs)
async def graph_search(query: str, reason: str) -> str:
    """Cypher-based graph traversal for structural/attribution queries (who/what/where).
    Use for structural queries like affiliations, sessions, references, topic connections, counts, traversals, and talkingpoint relationships.
    Not for content-only semantics—use semantic_search for meaning.
    Args:
    - query: Natural-language structural question to answer with Cypher
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
    - query: The query requiring both content and relationship reasoning.
    - node_label: The node label to use for the semantic component.
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

    # If a custom hybrid synthesis prompt is provided, use LLM to generate a single consolidated answer
    hybrid_prompt = _build_prompt_template(
        text_env="HYBRID_SYNTHESIS_PROMPT",
        file_env="HYBRID_SYNTHESIS_PROMPT_FILE",
        input_variables=["content_insights", "structural_relationships", "question"],
        label="hybrid synthesis",
    )
    if hybrid_prompt is not None:
        print("🧠 Performing hybrid synthesis with custom prompt...")
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
        try:
            prompt_text = hybrid_prompt.format(
                content_insights=str(semantic_results),
                structural_relationships=str(graph_results),
                question=actual_query,
            )
            llm = make_chat_model(qa_model, **qa_kwargs)
            response = await llm.ainvoke(prompt_text)
            return getattr(response, "content", str(response))
        except Exception as e:
            return f"Hybrid synthesis failed: {str(e)[:200]}"
    
    # Default behavior: return a combined report
    return f"""Hybrid analysis for: "{actual_query}"

📊 CONTENT INSIGHTS ({actual_node_label} nodes):
{semantic_results}

🔗 STRUCTURAL RELATIONSHIPS:
{graph_results}

✅ SYNTHESIS: Combining conference content with relationship patterns to provide comprehensive understanding of the science diplomacy discussion."""


# List of all available tools
TOOLS = [semantic_search, graph_search, hybrid_search]