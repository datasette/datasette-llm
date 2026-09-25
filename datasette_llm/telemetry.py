"""OpenTelemetry for datasette-llm: prompt spans and token usage, API-only."""

from contextlib import contextmanager
import contextvars
from importlib.metadata import PackageNotFoundError, version

import llm as llm_library
from opentelemetry import context, metrics, trace
from opentelemetry.trace import SpanKind, Status, StatusCode

try:
    _version = version("datasette-llm")
except PackageNotFoundError:  # pragma: no cover
    _version = None

tracer = trace.get_tracer("datasette_llm", _version)
meter = metrics.get_meter("datasette_llm", _version)

SPAN_PROMPT = "datasette_llm.prompt"
KIND = "datasette_llm.kind"
PURPOSE = "datasette_llm.purpose"
ACTOR_PRESENT = "datasette_llm.actor.present"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ERROR_TYPE = "error.type"
OUTCOME = "datasette_llm.outcome"
GEN_AI_TOKEN_TYPE = "gen_ai.token.type"
NO_PURPOSE = "none"
# llm.PauseChain (llm>=0.32a3) suspends a turn: an outcome, not an error
_NOT_ERRORS = getattr(llm_library, "PauseChain", ())
# Tells exit in our own Context apart from an abandoned chain's finalizer
_owner = contextvars.ContextVar("datasette_llm_prompt_span")

# The GenAI semantic conventions' recommended buckets for token counts
TOKEN_BUCKETS = (
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
)
token_usage = meter.create_histogram(
    "datasette_llm.token.usage",
    unit="{token}",
    description="Tokens per completed model response, by purpose",
    explicit_bucket_boundaries_advisory=TOKEN_BUCKETS,
)


def usage_recorder(purpose, model_id):
    "An on_done callback recording a finished response's reported token usage."
    purpose = NO_PURPOSE if purpose is None else purpose
    attributes = {PURPOSE: purpose, GEN_AI_REQUEST_MODEL: model_id}

    def record(response):
        for token_type in ("input", "output"):
            count = getattr(response, f"{token_type}_tokens", None)
            if count is not None:
                token_usage.record(count, {**attributes, GEN_AI_TOKEN_TYPE: token_type})

    return record


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
    previous = context.get_current()
    marker = _owner.set(None)
    context.attach(trace.set_span_in_context(span, previous))
    try:
        yield
    except GeneratorExit:
        # The consumer stopped iterating a chain early: not an error
        span.set_attribute(OUTCOME, "abandoned")
        raise
    except _NOT_ERRORS:
        raise
    except Exception as ex:
        span.set_attribute(ERROR_TYPE, type(ex).__qualname__)
        span.set_status(Status(StatusCode.ERROR))
        raise
    finally:
        # Not context.detach(), which logs an error when an abandoned chain
        # is finalized in another contextvars.Context. In our own Context,
        # restore the previous context (even over one a hook leaked); in a
        # finalizer's Context there is nothing of ours to restore.
        try:
            _owner.reset(marker)
        except ValueError:
            pass
        else:
            context.attach(previous)
        span.end()
