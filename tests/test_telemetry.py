import asyncio
from contextlib import asynccontextmanager, contextmanager
import gc
import json

from datasette import hookimpl
from datasette.app import Datasette
from datasette.plugins import pm
import llm
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
import pytest

from datasette_llm import LLM, telemetry

TOOL_PROMPT = json.dumps(
    {"tool_calls": [{"name": "example", "arguments": {"input": "x"}}], "prompt": "hi"}
)


def example(input: str) -> str:
    return f"Example output for {input}"


@pytest.fixture(scope="session")
def _exporter():
    # The global tracer provider can only be set once per process
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def exporter(_exporter):
    _exporter.clear()
    return _exporter


def spans(exporter, name="datasette_llm.prompt"):
    return [s for s in exporter.get_finished_spans() if s.name == name]


@contextmanager
def prompt_hook(on_enter):
    class Plugin:
        __name__ = "test_telemetry_hook"

        @hookimpl
        def llm_prompt_context(self, datasette, model_id, prompt, purpose):
            @asynccontextmanager
            async def ctx(result):
                on_enter()
                yield

            return ctx

    plugin = Plugin()
    pm.register(plugin)
    try:
        yield
    finally:
        pm.unregister(plugin)


def open_hook_span():
    trace.get_tracer("test").start_span("hook").end()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conversation,purpose,actor", [(False, "agent", None), (True, None, {"id": "a1"})]
)
@pytest.mark.parametrize("kind", ["prompt", "chain"])
async def test_prompt_and_chain_spans(exporter, conversation, purpose, actor, kind):
    llm_ = LLM(Datasette(memory=True))
    model = await llm_.model("echo", purpose=purpose, actor=actor)
    target = model.conversation() if conversation else model
    current = []
    with prompt_hook(open_hook_span):
        if kind == "prompt":
            await (await target.prompt("Hello")).text()
        else:
            chain = target.chain(TOOL_PROMPT, tools=[example])
            async for response in chain.responses():
                await response.text()
                current.append(trace.get_current_span().get_span_context())

    (span,) = spans(exporter)
    # Exactly these attributes: no actor id, no text
    assert dict(span.attributes) == {
        "datasette_llm.kind": kind,
        "datasette_llm.purpose": purpose or "none",
        "datasette_llm.actor.present": actor is not None,
        "gen_ai.request.model": "echo",
    }
    assert span.status.status_code == StatusCode.UNSET
    # Hook work nests under the span; a chain's span is current while iterating
    (hook,) = spans(exporter, "hook")
    assert hook.parent.span_id == span.context.span_id
    if kind == "chain":
        assert current == [span.context] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("close", [True, False])
async def test_chain_abandoned(exporter, caplog, close):
    model = await LLM(Datasette(memory=True)).model("echo")
    gen = model.chain(TOOL_PROMPT, tools=[example]).responses()
    async for response in gen:
        break
    if close:
        await gen.aclose()
        assert not trace.get_current_span().get_span_context().is_valid
    else:
        # asyncio finalizes the generator in another task and Context; the
        # span stays current in this task's context, by design (see README)
        del gen
        gc.collect()
        await asyncio.sleep(0.01)
    (span,) = spans(exporter)
    assert span.status.status_code == StatusCode.UNSET
    assert span.attributes["datasette_llm.outcome"] == "abandoned"
    assert "Failed to detach context" not in caplog.text


@pytest.mark.asyncio
async def test_hook_leaked_context_does_not_reach_caller(exporter):
    leaked = trace.get_tracer("test").start_span("leaked")
    attach = lambda: context.attach(trace.set_span_in_context(leaked))
    model = await LLM(Datasette(memory=True)).model("echo")
    with prompt_hook(attach):
        await model.prompt("Hello")
    assert trace.get_current_span() is not leaked


@pytest.mark.asyncio
async def test_no_sdk_attaches_nothing(exporter, monkeypatch):
    monkeypatch.setattr(telemetry, "tracer", trace.NoOpTracer())
    model = await LLM(Datasette(memory=True)).model("echo")
    before = context.get_current()
    async for response in model.chain(TOOL_PROMPT, tools=[example]).responses():
        assert context.get_current() == before
    assert spans(exporter) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["prompt", "chain"])
@pytest.mark.parametrize("exception", [ValueError, llm.PauseChain])
async def test_hook_exception(exporter, kind, exception):
    def boom():
        raise exception("secret message")

    model = await LLM(Datasette(memory=True)).model("echo", purpose="agent")
    with prompt_hook(boom), pytest.raises(exception):
        if kind == "prompt":
            await model.prompt("Hello")
        else:
            async for response in model.chain("Hello").responses():
                pass

    (span,) = spans(exporter)
    assert not span.events and not span.status.description
    if exception is ValueError:
        assert span.attributes["error.type"] == "ValueError"
        assert span.status.status_code == StatusCode.ERROR
    else:
        # PauseChain suspends a turn; it is not an error
        assert "error.type" not in span.attributes
        assert span.status.status_code == StatusCode.UNSET
