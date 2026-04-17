"""
OpenTelemetry distributed tracing setup.
Exports traces to Jaeger (or OTLP-compatible backend).
"""
import logging
import os

logger = logging.getLogger(__name__)


def setup_tracing(service_name: str = "vortexvision-api"):
    """Initialize OpenTelemetry tracing. No-ops gracefully if deps not installed."""
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name, "service.version": "1.0.0"})
        provider = TracerProvider(resource=resource)

        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        FastAPIInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()

        logger.info("OpenTelemetry tracing initialized → %s", otlp_endpoint)
    except ImportError:
        logger.info("OpenTelemetry not installed — tracing disabled")
