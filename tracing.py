"""Phoenix tracing setup using OpenTelemetry + OpenInference.

This module configures an OTLP HTTP exporter pointing at a Phoenix server
and auto-instruments LangChain and OpenAI so LLM calls, tools, and chains
are captured as traces.

Environment variables:
- PHOENIX_OTLP_ENDPOINT: OTLP HTTP endpoint for Phoenix (default: http://localhost:6006/v1/traces)
- SERVICE_NAME: logical service name to appear in Phoenix (default: neo4jchat-langgraph)
"""
from __future__ import annotations

import os
from typing import Optional, Dict

_initialized = False


def _normalize_endpoint(endpoint: str) -> str:
    # Ensure the exporter endpoint ends with /v1/traces for OTLP HTTP (self-hosted)
    if endpoint.endswith("/v1/traces"):
        return endpoint
    # Strip trailing slashes and append the path
    return endpoint.rstrip("/") + "/v1/traces"


def _parse_headers(header_str: str) -> Dict[str, str]:
    """Parse a comma-separated header string like "k=v,k2=v2" into a dict."""
    headers: Dict[str, str] = {}
    parts = [p.strip() for p in header_str.split(",") if p.strip()]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            headers[k.strip()] = v.strip()
    return headers


def setup_tracing(service_name: Optional[str] = None, otlp_endpoint: Optional[str] = None) -> None:
    """Initialize tracing for Phoenix Cloud or self-hosted.

    Preference order:
    1) If arize-phoenix-otel is available, use phoenix.otel.register (works with Cloud or self-hosted via env)
    2) Else, fallback to manual OTLP HTTP exporter using PHOENIX_OTLP_ENDPOINT

    Safe to call multiple times; initialization occurs only once.
    """
    global _initialized
    if _initialized:
        return

    svc_name = service_name or os.getenv("SERVICE_NAME", "neo4jchat-langgraph")

    # Try Phoenix OTEL helper first (best for Phoenix Cloud)
    try:
        from phoenix.otel import register  # type: ignore

        project_name = os.getenv("PHOENIX_PROJECT_NAME", svc_name)
        # Env vars consumed by phoenix.otel.register:
        # - PHOENIX_COLLECTOR_ENDPOINT (Cloud hostname or self-hosted OTLP URL)
        # - PHOENIX_API_KEY (Cloud)
        # - PHOENIX_CLIENT_HEADERS (legacy Cloud instances)
        _ = register(project_name=project_name, auto_instrument=True)

        _initialized = True
        print(f"[Phoenix tracing] initialized via phoenix.otel.register (project={project_name})")
        return
    except Exception as e:
        print(f"[Phoenix tracing] phoenix.otel.register not used: {e}")

    # Fallback: manual OTLP HTTP exporter (self-hosted or generic OTLP)
    endpoint = otlp_endpoint or os.getenv("PHOENIX_OTLP_ENDPOINT", "http://localhost:6006/v1/traces")
    endpoint = _normalize_endpoint(endpoint)

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        # Optional: headers support (legacy Cloud instances)
        headers_env = os.getenv("PHOENIX_CLIENT_HEADERS") or os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
        headers = _parse_headers(headers_env) if headers_env else None

        # Configure tracer provider with service name
        resource = Resource.create({"service.name": svc_name})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        # Configure OTLP HTTP exporter to Phoenix
        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        # Auto-instrument LangChain and OpenAI via OpenInference
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor

            LangChainInstrumentor().instrument(tracer_provider=provider)
        except Exception as e:
            print(f"[Phoenix tracing] LangChain instrumentation skipped: {e}")

        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().instrument()
        except Exception as e:
            print(f"[Phoenix tracing] OpenAI instrumentation skipped: {e}")

        _initialized = True
        print(f"[Phoenix tracing] initialized (service.name={svc_name}, endpoint={endpoint})")
    except Exception as e:
        # Never break the app if tracing fails; just log and continue.
        print(f"[Phoenix tracing] setup skipped due to error: {e}")

