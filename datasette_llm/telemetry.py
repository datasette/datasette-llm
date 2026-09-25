"""OpenTelemetry for datasette-llm: a span per prompt()/chain(), API-only."""

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

import llm as llm_library
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

try:
    _version = version("datasette-llm")
except PackageNotFoundError:  # pragma: no cover
    _version = None

tracer = trace.get_tracer("datasette_llm", _version)

SPAN_PROMPT = "datasette_llm.prompt"
KIND = "datasette_llm.kind"
PURPOSE = "datasette_llm.purpose"
ACTOR_PRESENT = "datasette_llm.actor.present"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ERROR_TYPE = "error.type"
NO_PURPOSE = "none"
# llm.PauseChain (llm>=0.32a3) suspends a turn: an outcome, not an error
_NOT_ERRORS = getattr(llm_library, "PauseChain", ())


@contextmanager
def prompt_span(kind, purpose, model_id, actor):
    "Current datasette_llm.prompt span; no text, actor ids or error messages."
    span = tracer.start_span(
        SPAN_PROMPT,
        kind=SpanKind.INTERNAL,
        attributes={
            KIND: kind,
            PURPOSE: NO_PURPOSE if purpose is None else purpose,
            GEN_AI_REQUEST_MODEL: model_id,
            ACTOR_PRESENT: actor is not None,
        },
    )
    if not span.is_recording():
        # No SDK: don't attach a context an abandoned chain could leak
        yield
        return
    with trace.use_span(
        span, end_on_exit=True, record_exception=False, set_status_on_exception=False
    ):
        try:
            yield
        except _NOT_ERRORS:
            raise
        except Exception as ex:
            span.set_attribute(ERROR_TYPE, type(ex).__qualname__)
            span.set_status(Status(StatusCode.ERROR))
            raise
