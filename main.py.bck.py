"""Main entry point for the Neo4j LangGraph chatbot."""

import os
import argparse
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Load env early from this directory to avoid CWD issues
try:
    load_dotenv(Path(__file__).with_name(".env"))
except Exception as _e:
    # Fallback to default behavior
    try:
        load_dotenv()
    except Exception:
        pass

# Initialize tracing (Phoenix via OpenTelemetry) before importing graph/components
try:
    from tracing import setup_tracing
    setup_tracing()
except Exception as _e:
    # Tracing is optional; proceed if unavailable
    print(f"[Phoenix tracing] not initialized: {_e}")

from langchain_core.messages import HumanMessage

from graph import graph


def create_embeddings_legacy(node_label, text_properties, index_name="embeddings", embedding_property="embeddings"):
    """Legacy embedding creation function for backward compatibility."""
    from neo4j import GraphDatabase
    from langchain_community.vectorstores import Neo4jVector
    from langchain_openai import OpenAIEmbeddings
    
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    print(f"🔧 Creating embeddings for {node_label} nodes...")
    
    # Validate text properties format
    for prop in text_properties:
        if "." not in prop:
            raise ValueError(f"Invalid format '{prop}'. Expected format: Node.Property")
        prop_node, _ = prop.split(".", 1)
        if prop_node != node_label:
            raise ValueError(f"Property '{prop}' doesn't match node label '{node_label}'")
    
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        
        # Create Neo4jVector for existing graph
        vector_store = Neo4jVector.from_existing_graph(
            embedding=OpenAIEmbeddings(),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=index_name,
            node_label=node_label,
            text_node_properties=[prop.split(".", 1)[1] for prop in text_properties],
            embedding_node_property=embedding_property
        )
        
        print(f"✅ Successfully created embeddings for {node_label} nodes")
        print(f"   Properties: {', '.join(text_properties)}")
        print(f"   Index: {index_name}")
        print(f"   Embedding property: {embedding_property}")
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to create embeddings: {str(e)}")
        return False


async def run_query(query: str, model: str = "gpt-4", temperature: float | None = None, system_prompt: str = None):
    """Run a single query through the agent."""
    try:
        if system_prompt is None:
            system_prompt = """#Objective:
Produce an accurate answer to the user's question by retrieving and synthesizing only the necessary evidence from one or both of the following sources: the Neo4j conference knowledge graph and the vectorized corpora of Answer, Question, and Talkingpoint texts. Present the final result clearly in the required ReAct output.

You analyze a curated Science Diplomacy conference knowledge graph and its vector indexes.

#Routing:
GRAPH_SEARCH → structural questions (who/what/where), affiliations, sessions, references, topics, counts, traversals.
SEMANTIC_SEARCH → content-only meaning over vectorized text. Specify node_label: Answer|Question|Talkingpoint.
HYBRID_SEARCH → "who said what about X" or any mix of content + structure. Specify node_label for semantic portion.

#Hard rules:
Use only information in the graph or vector indexes; do not invent labels, properties, nodes, or facts.
If nothing relevant is found, say so explicitly.

#Ontology (labels, key properties, directions):

##Entities: 
Person(name, role); Organisation(name, acronym); Initiative(name, type); Place(name, type); Topic(name); Session(name, segment_name, start_time, end_time); Question(text, timestamp_start, timestamp_end); Answer(text, timestamp_start, timestamp_end); Talkingpoint(text, resolved_text, category ∈ {Challenge, Proposal, Observation, Opportunity, Principle})

##Relationships (directed): 
Person-IS_AFFILIATED_WITH->Organisation; Person-ASKS->Question; Person-ANSWERS->Answer; Question-PROMPTS->Answer; Question-IN_SESSION->Session; Answer-IN_SESSION->Session; Answer-CONTAINS->Talkingpoint; Talkingpoint-ABOUT->Topic; Initiative-ADDRESSES->Topic; Initiative-HAS_GEOGRAPHIC_FOCUS->Place; Talkingpoint-HAS_PLACE_SCOPE->Place; Talkingpoint-(IS_SOLUTION_TO|CAUSES|SUPPORTS|REFUTES|ENABLES|CONTRASTS_WITH)->Talkingpoint; (Question|Answer)-REFERENCES->(Person|Organisation|Initiative|Place) with intent ∈ {mentions, cites, builds_upon, acknowledges, cites_as_example, locates}.

#Decision:
Relationships, attribution, or counts → GRAPH_SEARCH.
Content/themes only → SEMANTIC_SEARCH with appropriate node_label (Answer|Question|Talkingpoint).
Both content and attribution/structure → HYBRID_SEARCH with appropriate node_label.

#Final Answer Generation:
When you have gathered all necessary evidence from your tools, you must synthesize it into a final answer. Follow these principles:

1.  **Strictly Ground in Evidence:** Every statement in the final answer must be directly traceable to the information retrieved. Do not introduce any outside information, assumptions, or embellishments, no matter how plausible. If the evidence is incomplete, state that clearly rather than filling in the gaps.
2.  **Synthesize, Don't Just List:** Do not simply repeat the raw output from the `Observation` steps. Integrate the findings from both semantic and graph searches into a cohesive, narrative answer.
3.  **Provide Insight, Not Just Data:** Move beyond data retrieval to data interpretation. Explain the *significance* of the findings. Connect disparate pieces of information to reveal patterns, contrasts, or key takeaways. For example, instead of "Person A mentioned Topic X," aim for "Person A, representing Organization Y, proposed a solution for Topic X, which directly contrasts with the challenge raised by Person B."
4.  **Be Precise and Specific:** Use the exact names of `People`, `Organisations`, `Initiatives`, and `Topics`. Weave the meaning of relationships into your narrative instead of stating them verbatim (e.g., write "This proposal addresses the challenge of..." instead of "...`IS_SOLUTION_TO`...").
5.  **Stay Focused:** Ensure the entire answer is a direct response to the user's original question. Eliminate any information that, while interesting, does not contribute to answering the core query.
6.  **Structure for Clarity:** Use short paragraphs for explanations and bullet points for lists (e.g., a list of people who discussed a topic).

For semantic/hybrid tools, use JSON like {"query": "<short query>", "node_label": "Answer"} as input.

If not enough evidence was found to directly and reliably answer the question, state "No sufficent matching data was found to reliably answer the question." """
        
        configurable = {
            "model": model,
            "system_prompt": system_prompt,
        }
        if temperature is not None:
            configurable["temperature"] = float(temperature)

        config = {
            "configurable": configurable
        }
        
        response = await graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config
        )
        return response["messages"][-1].content
    except Exception as e:
        return f"Error: {str(e)}"


async def interactive_mode(model: str = "gpt-4", temperature: float | None = None):
    """Run interactive chatbot mode."""
    print("Neo4j LangGraph Chatbot is ready. Enter your query (or 'exit' to quit).")
    print("The agent will intelligently choose between semantic_search, graph_search, or hybrid_search.")
    
    while True:
        query = input("> ")
        if query.lower() == 'exit':
            break
            
        response = await run_query(query, model, temperature)
        print(f"\n{response}\n")


def _startup_log(agent_model: str) -> None:
    """Print effective configuration and check Neo4j connectivity."""
    try:
        cypher_model = os.getenv("CYPHER_LLM_MODEL", "openai/gpt-4")
        qa_model = os.getenv("QA_LLM_MODEL", cypher_model)
        print("Startup configuration:")
        print(f"- Agent model: {agent_model}")
        print(f"- CYPHER_LLM_MODEL: {cypher_model}")
        print(f"- QA_LLM_MODEL: {qa_model}")
        # Quick Neo4j connectivity check
        try:
            from langchain_neo4j import Neo4jGraph  # type: ignore
            g = Neo4jGraph(
                url=os.getenv("NEO4J_URI"),
                username=os.getenv("NEO4J_USERNAME"),
                password=os.getenv("NEO4J_PASSWORD"),
                database=os.getenv("NEO4J_DATABASE", "neo4j"),
            )
            g.refresh_schema()
            print("- Neo4j: connected")
        except Exception as e:
            print(f"- Neo4j: connection failed: {str(e)[:200]}")
    except Exception as e:
        print(f"[startup log] error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Science Diplomacy Conference Knowledge Graph Chatbot (LangGraph)")
    parser.add_argument("--create-embeddings", action="store_true", help="Create embeddings for specified node type")
    parser.add_argument("--node-label", type=str, help="Node label for embedding creation (Question, Answer, Talkingpoint)")
    parser.add_argument("--text-properties", nargs="+", help="Properties to embed (Node.Property format)")
    parser.add_argument("--index-name", type=str, default="embeddings", help="Vector index name")
    parser.add_argument("--embedding-property", type=str, default="embeddings", help="Property to store embeddings")
    parser.add_argument("--test", action="store_true", help="Run a test query instead of interactive mode")
    parser.add_argument("--query", type=str, default="What are the main topics discussed?", help="Query to run in test mode")
    import os as _os
    parser.add_argument(
        "--model",
        type=str,
        default=_os.getenv("AGENT_LLM_MODEL", "openai/gpt-4"),
        help="Model to use for the agent (can be provider-prefixed)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Agent model temperature. If omitted, uses AGENT_TEMPERATURE env; if unset, not sent to model.",
    )
    args = parser.parse_args()

    # Print startup configuration and connectivity check
    _startup_log(args.model)

    # Handle embedding creation
    if args.create_embeddings:
        if not args.node_label or not args.text_properties:
            print("❌ Both --node-label and --text-properties must be provided for embedding creation")
            return
        
        print(f"Creating embeddings for {args.node_label} nodes...")
        success = create_embeddings_legacy(
            node_label=args.node_label,
            text_properties=args.text_properties,
            index_name=args.index_name,
            embedding_property=args.embedding_property
        )
        
        if success:
            print(f"\n✅ Embeddings created successfully!")
            print(f"You can now use semantic search with node type: {args.node_label}")
        return

    # Run in test mode or interactive mode
    # Resolve final temperature: CLI overrides env; env supports '', 'null', 'None' as unset
    env_temp = _os.getenv("AGENT_TEMPERATURE")
    temp_from_env = None
    if env_temp is not None and env_temp.strip().lower() not in ("", "none", "null"):
        try:
            temp_from_env = float(env_temp)
        except ValueError:
            temp_from_env = None
    effective_temp = args.temperature if args.temperature is not None else temp_from_env

    if args.test:
        print(f"Running test query: {args.query}")
        response = asyncio.run(run_query(args.query, args.model, effective_temp))
        print(f"Response: {response}")
    else:
        asyncio.run(interactive_mode(args.model, effective_temp))


if __name__ == "__main__":
    main()