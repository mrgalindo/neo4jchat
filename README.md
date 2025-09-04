# Neo4j LangGraph Chatbot

A streamlined implementation of the Science Diplomacy Conference Knowledge Graph chatbot using LangGraph with ReAct pattern.

## Architecture

This refactored version uses LangGraph to implement a ReAct (Reasoning and Acting) agent that intelligently selects between three search modalities:

- **semantic_search**: Content-focused queries using vector embeddings
- **graph_search**: Structural queries using Cypher traversal  
- **hybrid_search**: Complex queries combining both approaches

## Key Components

- `state.py`: Defines the agent state structure
- `context.py`: Configuration and runtime context
- `tools.py`: Search tools (semantic, graph, hybrid)
- `graph.py`: LangGraph ReAct agent implementation
- `main.py`: Application entry point

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and set your credentials:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Interactive Mode
```bash
python main.py
```

### Test Mode
```bash
python main.py --test --query "What are the main challenges in science diplomacy?"
```

### Create Embeddings
```bash
python main.py --create-embeddings --node-label Answer --text-properties Answer.text
```

## Key Improvements

1. **Streamlined Architecture**: Uses LangGraph's native ReAct pattern
2. **Tool-based Design**: Each search type is a proper tool
3. **Async Support**: Full async/await implementation
4. **Better Error Handling**: Graceful fallbacks and error messages
5. **Cleaner Configuration**: Centralized context management
6. **Maintainable Code**: Modular structure with clear separation of concerns

## Multi-provider and model configuration

This project supports multiple LLM providers and separate models for different tasks.

- Provider-prefixed model names: `openai/gpt-4o`, `anthropic/claude-3-opus-20240229`, `groq/llama3-70b-8192`, `google/gemini-1.5-pro`
- Default provider when the model has no prefix: set `LLM_PROVIDER` (openai|anthropic|groq|google)
- Separate models per function:
  - `CYPHER_LLM_MODEL`: Used for generating Cypher queries (GraphCypherQAChain cypher_llm)
  - `QA_LLM_MODEL`: Used for QA summarization and semantic RetrievalQA (GraphCypherQAChain qa_llm and semantic tool). If not set, falls back to `CYPHER_LLM_MODEL`.
  - Agent model: pass via `--model` on CLI; if omitted, you can set `AGENT_LLM_MODEL` in your environment.

Example env (.env):

```
LLM_PROVIDER=openai
CYPHER_LLM_MODEL=openai/gpt-4o
QA_LLM_MODEL=anthropic/claude-3-opus-20240229
# Optionally, set a default agent model too:
# AGENT_LLM_MODEL=groq/llama3-70b-8192
```

Examples:

- Run test with agent model specified on CLI (overrides AGENT_LLM_MODEL):
```bash
python main.py --test --query "Who discussed AI governance?" --model anthropic/claude-3-opus-20240229
```
- Default provider via env and plain model name for tools:
```bash
export LLM_PROVIDER=groq
export CYPHER_LLM_MODEL=llama3-70b-8192
export QA_LLM_MODEL=llama3-8b-8192
python main.py --test --query "Top initiatives on ocean policy?" --model llama3-8b-8192
```

## Tracing (Open Source) with Arize Phoenix

Phoenix tracing is wired into the app using OpenTelemetry + OpenInference. It captures LangChain chains, LLM calls, tool executions, and related metadata.

Setup steps:

1. Install dependencies (already listed in requirements.txt):
   - arize-phoenix
   - openinference-instrumentation-langchain
   - openinference-instrumentation-openai
   - opentelemetry-sdk
   - opentelemetry-exporter-otlp

2. Start a Phoenix server locally (open source). Phoenix accepts traces over OTLP HTTP. Refer to the official docs for self-hosting and quickstart guidance:
   - https://arize.com/docs/phoenix/tracing/llm-traces

   By default, this project exports to `http://localhost:6006/v1/traces`. If your Phoenix endpoint differs, set:
   - `PHOENIX_OTLP_ENDPOINT=http://<host>:<port>/v1/traces`

   Optional:
   - `SERVICE_NAME=neo4jchat-langgraph` (overrides the default service name shown in Phoenix)

3. Run the app to generate traces:
```bash
python main.py --test --query "What are the main challenges in science diplomacy?"
```

4. Open the Phoenix UI to inspect traces, spans, token usage, parameters, and errors.

Notes:
- Tracing initialization is best-effort and won’t break the app if Phoenix isn’t running.
- You can change `PHOENIX_OTLP_ENDPOINT` without code changes.
- The tracing setup lives in `tracing.py` and is initialized early in `main.py`.
