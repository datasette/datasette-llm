import json
from contextlib import asynccontextmanager

import llm
import pytest
from datasette import hookimpl
from datasette.app import Datasette
from datasette.plugins import pm
from datasette_secrets import get_secrets

from datasette_llm import LLM, ModelConfigurationError, ModelNotFoundError


def make_llm(**config):
    return LLM(Datasette(memory=True, config={"plugins": {"datasette-llm": config}}))


def endpoint(model="local", **kwargs):
    return {"model": model, "api_base": "https://endpoint.test/prefix/v1", **kwargs}


def store_keys(**keys):
    path = llm.user_dir() / "keys.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keys))


def payload(responses, text="Hello", tool=False):
    if responses:
        output = (
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "double",
                "arguments": '{"number": 3}',
                "status": "completed",
            }
            if tool
            else {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
        return {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "remote",
            "output": [output],
            "status": "completed",
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }
    message = {"role": "assistant", "content": None if tool else text}
    if tool:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "double", "arguments": '{"number": 3}'},
            }
        ]
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "created": 1,
        "model": "remote",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool else "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def add_response(mock, responses, stream=False, tool=False, text="Hello", base=None):
    url = (base or endpoint()["api_base"]) + (
        "/responses" if responses else "/chat/completions"
    )
    body = payload(responses, text=text, tool=tool)
    if not stream:
        mock.add_response(method="POST", url=url, json=body)
        return
    if responses:
        item = body["output"][0]
        events = [
            {"type": "response.output_item.added", "item": item, "output_index": 0}
        ]
        if not tool:
            events.append({"type": "response.output_text.delta", "delta": text})
        events.extend(
            [
                {"type": "response.output_item.done", "item": item, "output_index": 0},
                {"type": "response.completed", "response": body},
            ]
        )
    else:
        message = body["choices"][0]["message"]
        if tool:
            message["tool_calls"][0]["index"] = 0
        events = [
            {
                "id": "chatcmpl_1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "remote",
                "usage": body["usage"],
                "choices": [
                    {
                        "index": 0,
                        "delta": message,
                        "finish_reason": "tool_calls" if tool else "stop",
                    }
                ],
            }
        ]
    content = (
        "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        + "data: [DONE]\n\n"
    )
    mock.add_response(
        method="POST",
        url=url,
        content=content,
        headers={"Content-Type": "text/event-stream"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_endpoint_prompt(httpx2_mock, monkeypatch, responses, stream):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-send-env-key")
    monkeypatch.setenv("DATASETTE_SECRETS_OPENAI_API_KEY", "do-not-send-secret")
    store_keys(openai="do-not-send-stored-key")
    add_response(httpx2_mock, responses, stream)
    ref = endpoint(
        model_name="remote", responses=responses, headers={"X-App": "Datasette"}
    )
    model = await make_llm(default_model=ref).model()
    response = await model.prompt("Hi", stream=stream)
    assert await response.text() == "Hello"
    usage = await response.usage()
    assert (usage.input, usage.output) == (3, 2)
    request = httpx2_mock.get_requests()[0]
    assert request.headers["authorization"] == "Bearer DUMMY_KEY"
    assert request.headers["x-app"] == "Datasette"
    body = json.loads(request.content)
    assert body["model"] == "remote"
    assert body["stream"] is stream
    assert "reasoning" not in body
    assert model.model_id == "local"


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_endpoint_tool_chain(httpx2_mock, responses, stream):
    add_response(httpx2_mock, responses, stream, tool=True)
    add_response(httpx2_mock, responses, stream, text="The answer is 6")
    calls, observed = [], []

    def double(number: int) -> int:
        calls.append(number)
        return number * 2

    class Observer:
        @hookimpl
        def llm_prompt_context(self, model_id, purpose, actor):
            @asynccontextmanager
            async def observe(result):
                assert (model_id, purpose, actor) == ("local", "agent", {"id": "alice"})
                yield

                async def done(response):
                    observed.append(await response.text())

                await result.on_response_done(done)

            return observe

    plugin = Observer()
    pm.register(plugin)
    try:
        model = await make_llm(default_model=endpoint(responses=responses)).model(
            purpose="agent", actor={"id": "alice"}
        )
        chain = model.chain("Double 3", tools=[double], stream=stream)
        replies = [reply async for reply in chain.responses()]
        assert await replies[-1].text() == "The answer is 6"
    finally:
        pm.unregister(plugin)
    assert calls == [3]
    assert observed == ["", "The answer is 6"]
    first, second = [json.loads(r.content) for r in httpx2_mock.get_requests()]
    assert first["tools"][0]["type"] == "function"
    if responses:
        assert {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "6",
        } in second["input"]
    else:
        assert {"role": "tool", "tool_call_id": "call_1", "content": "6"} in second[
            "messages"
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    [
        "default",
        "global-list",
        "purpose-default",
        "purpose-list",
        "inline",
        "inline-group",
    ],
)
@pytest.mark.parametrize("secret_override", [False, True])
async def test_endpoint_named_keys(httpx2_mock, monkeypatch, location, secret_override):
    store_keys(openrouter="stored-value", openai="do-not-send")
    if secret_override:
        monkeypatch.setenv("DATASETTE_SECRETS_openrouter", "secret-value")
    ref = endpoint(key="openrouter")
    configs = {
        "default": {"default_model": ref},
        "global-list": {"models": [ref]},
        "purpose-default": {"purposes": {"agent": {"model": ref}}},
        "purpose-list": {"purposes": {"agent": {"models": [ref]}}},
        "inline": {},
        "inline-group": {},
    }
    client = make_llm(**configs[location])
    add_response(httpx2_mock, False)
    purpose = "agent" if location.startswith("purpose") else None
    if location == "inline-group":
        async with client.group(ref) as model:
            response = await model.prompt("Hi", stream=False)
    else:
        model = await client.model(
            ref if location == "inline" else "local", purpose=purpose
        )
        response = await model.prompt("Hi", stream=False)
    assert await response.text() == "Hello"
    expected = "secret-value" if secret_override else "stored-value"
    assert (
        httpx2_mock.get_requests()[0].headers["authorization"] == f"Bearer {expected}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("key", [None, "openai"])
async def test_responses_chat_fallback_credentials(httpx2_mock, responses, key):
    store_keys(openai="chosen-stored-key")
    add_response(httpx2_mock, False)
    ref = endpoint(responses=responses, **({"key": key} if key else {}))
    model = await make_llm(default_model=ref).model()
    response = await model.prompt(
        "Hi", stream=False, **({"chat_completions": True} if responses else {})
    )
    assert await response.text() == "Hello"
    expected = "chosen-stored-key" if key else "DUMMY_KEY"
    assert (
        httpx2_mock.get_requests()[0].headers["authorization"] == f"Bearer {expected}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("require_keys", [False, True])
async def test_missing_named_key(monkeypatch, require_keys):
    store_keys(openai="wrong-provider-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-env-key")
    ref = endpoint(key="missing")
    client = make_llm(models=[ref], require_keys=require_keys)
    assert [m.model_id for m in await client.models()] == (
        [] if require_keys else ["local"]
    )
    with pytest.raises(llm.NeedsKeyException, match="missing.*not found"):
        await client.model("local")


@pytest.mark.asyncio
async def test_endpoint_key_availability_and_order():
    store_keys(available="stored")
    client = make_llm(
        default_model=endpoint("default"),
        models=[
            endpoint("missing", key="missing"),
            endpoint("keyed", key="available"),
            "echo",
        ],
        purposes={
            "agent": {
                "model": endpoint("purpose-default"),
                "models": [endpoint("purpose")],
            }
        },
    )
    assert [m.model_id for m in await client.models()] == ["default", "keyed", "echo"]
    assert [m.model_id for m in await client.models(purpose="agent")] == [
        "purpose-default",
        "purpose",
        "keyed",
        "echo",
        "default",
    ]
    with pytest.raises(ModelNotFoundError):
        await client.model("purpose")
    assert (await client.model(purpose="agent")).model_id == "purpose-default"


@pytest.mark.asyncio
async def test_same_provider_different_named_keys():
    store_keys(available="stored", openai="provider-key")
    client = make_llm(
        models=[
            {"model": "gpt-4o", "key": "missing"},
            {"model": "gpt-4o-mini", "key": "available"},
        ]
    )
    assert [m.model_id for m in await client.models()] == ["gpt-4o-mini"]
    assert (await client.model("gpt-4o-mini"))._key == "stored"


@pytest.mark.asyncio
async def test_global_model_list_registers_secrets():
    client = make_llm(models=[endpoint(key="ENDPOINT_KEY")])
    names = [secret.name for secret in await get_secrets(client._datasette)]
    assert names.count("ENDPOINT_KEY") == 1
    assert len(names) == len(set(names))


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("mode", ["model", "conversation", "group"])
async def test_endpoint_options_and_conversations(httpx2_mock, responses, mode):
    add_response(httpx2_mock, responses)
    add_response(httpx2_mock, responses, text="Again")
    client = make_llm(
        default_model=endpoint(responses=responses, options={"temperature": 0.2})
    )

    async def prompt_twice(model):
        response = await model.prompt("Hi", stream=False)
        assert await response.text() == "Hello"
        response = await model.prompt("Again", stream=False, temperature=0.7)
        assert await response.text() == "Again"

    if mode == "group":
        async with client.group() as model:
            await prompt_twice(model)
            assert len(model._group._responses) == 2
    else:
        model = await client.model()
        await prompt_twice(model.conversation() if mode == "conversation" else model)
    first, second = [json.loads(r.content) for r in httpx2_mock.get_requests()]
    assert (first["temperature"], second["temperature"]) == (0.2, 0.7)
    if mode == "conversation":
        assert "Hello" in json.dumps(second["input" if responses else "messages"])


@pytest.mark.asyncio
async def test_purpose_and_inline_overrides(httpx2_mock):
    store_keys(
        global_key="global-value",
        purpose_key="purpose-value",
        inline_key="other_alias",
        other_alias="do-not-substitute",
    )
    client = make_llm(
        models=[endpoint(key="global_key", options={"temperature": 0.1})],
        purposes={
            "agent": {
                "model": {
                    "model": "local",
                    "key": "purpose_key",
                    "options": {"temperature": 0.2},
                }
            }
        },
    )
    for key, temperature, ref in [
        ("purpose-value", 0.2, "local"),
        (
            "other_alias",
            0.3,
            {"model": "local", "key": "inline_key", "options": {"temperature": 0.3}},
        ),
    ]:
        add_response(httpx2_mock, False)
        model = await client.model(ref, purpose="agent")
        response = await model.prompt("Hi", stream=False)
        await response.text()
        request = httpx2_mock.get_requests()[-1]
        assert request.headers["authorization"] == f"Bearer {key}"
        assert json.loads(request.content)["temperature"] == temperature


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,restriction",
    [
        (source, restriction)
        for source in ("explicit", "default", "purpose", "hook", "inline", "group")
        for restriction in ("global-block", "purpose-block", "allowlist", "filter-hook")
        if not (source in ("default", "purpose") and restriction == "allowlist")
    ],
)
async def test_selection_respects_filters(source, restriction):
    ref = endpoint()
    config = {"models": [ref]}
    if source == "default":
        config["default_model"] = ref
    if source == "purpose":
        config["purposes"] = {"agent": {"model": ref}}
    if restriction == "global-block":
        config["blocked_models"] = ["local"]
    elif restriction == "purpose-block":
        config.setdefault("purposes", {}).setdefault("agent", {})["blocked_models"] = [
            "local"
        ]
    elif restriction == "allowlist":
        config = {"models": ["echo"]}

    class Policy:
        @hookimpl
        async def llm_filter_models(self, models):
            if restriction == "filter-hook":
                return [m for m in models if m.model_id != "local"]

        @hookimpl
        def llm_default_model(self):
            if source == "hook":
                return "local"

    plugin = Policy()
    pm.register(plugin)
    try:
        client = make_llm(**config)
        argument = ref if source == "inline" or restriction == "allowlist" else "local"
        if source in ("default", "purpose", "hook"):
            argument = None
        with pytest.raises(ModelNotFoundError):
            if source == "group":
                async with client.group(argument, purpose="agent"):
                    pytest.fail("Blocked group was allowed")
            else:
                await client.model(argument, purpose="agent")
    finally:
        pm.unregister(plugin)


@pytest.mark.asyncio
async def test_endpoint_default_hook_and_instance_isolation():
    class Default:
        @hookimpl
        def llm_default_model(self):
            return "local"

    first = make_llm(models=[endpoint()])
    second = make_llm(models=[endpoint(api_base="https://second.test/v1")])
    plugin = Default()
    pm.register(plugin)
    try:
        assert (await first.model())._model.api_base == endpoint()["api_base"]
        assert (await second.model())._model.api_base == "https://second.test/v1"
    finally:
        pm.unregister(plugin)
    inline = await make_llm().model(endpoint("temporary"))
    assert inline.model_id == "temporary"
    with pytest.raises(llm.UnknownModelError):
        llm.get_async_model("local")
    with pytest.raises(llm.UnknownModelError):
        llm.get_async_model("temporary")
    with pytest.raises(ModelNotFoundError):
        await first.model("temporary")


@pytest.mark.asyncio
async def test_endpoint_actor_credentials_are_not_shared(httpx2_mock, monkeypatch):
    async def actor_secret(datasette, name, actor_id=None):
        if name == "actor-key":
            return f"key-for-{actor_id}"

    monkeypatch.setattr("datasette_llm.get_secret", actor_secret)
    client = make_llm(default_model=endpoint(key="actor-key"))
    alice = await client.model(actor={"id": "alice"})
    bob = await client.model(actor={"id": "bob"})
    assert alice._model is not bob._model
    for model in (alice, bob):
        add_response(httpx2_mock, False)
        response = await model.prompt("Hi", stream=False)
        await response.text()
    assert [r.headers["authorization"] for r in httpx2_mock.get_requests()] == [
        "Bearer key-for-alice",
        "Bearer key-for-bob",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"api_base": "https://different.test/v1"},
        {"responses": True},
        {"model_name": "other"},
        {"supports_tools": False},
    ],
)
async def test_conflicting_endpoint_ids(change):
    client = make_llm(
        default_model=endpoint(), purposes={"agent": {"model": endpoint(**change)}}
    )
    assert "local" not in {model.model_id for model in await client.models()}
    assert (await client.model("echo")).model_id == "echo"
    with pytest.raises(ModelConfigurationError, match="Conflicting.*local"):
        await client.model()
    with pytest.raises(ModelConfigurationError, match="Conflicting.*local"):
        await make_llm(models=[endpoint()]).model(endpoint(**change))


@pytest.mark.asyncio
async def test_repeated_endpoint_definition():
    ref = endpoint(model_name="remote")
    client = make_llm(
        default_model=ref,
        models=[ref],
        purposes={"agent": {"model": ref, "models": [ref]}},
    )
    assert [m.model_id for m in await client.models(purpose="agent")] == ["local"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", ["echo", "gpt-4o", "4o"])
async def test_endpoint_installed_model_collision(model_id):
    with pytest.raises(
        ModelConfigurationError, match="conflicts with an installed model or alias"
    ):
        await make_llm(default_model=endpoint(model_id)).model()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ref",
    [
        {},
        {"model": ""},
        {"model": "local", "options": []},
        {"model": "local", "responses": True},
        endpoint(responses="yes"),
        endpoint(key=""),
        endpoint(api_base="ftp://endpoint.test"),
        endpoint(api_base="https://endpoint.test:invalid"),
        endpoint(api_base="https://user:password@endpoint.test/v1"),
        endpoint(api_base="https://endpoint.test/v1?key=secret"),
        endpoint(api_base="https://endpoint.test/v1#responses"),
        endpoint(unknown=True),
        endpoint(headers={"X-App": 1}),
    ],
)
async def test_invalid_endpoint_reference(ref):
    with pytest.raises(ModelConfigurationError):
        await make_llm(default_model=ref, models=[ref]).models()


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("mode", ["model", "conversation", "chain"])
async def test_minimal_endpoint_reference_and_streaming_disabled(
    httpx2_mock, responses, mode
):
    add_response(httpx2_mock, responses)
    model = await make_llm().model(endpoint(can_stream=False, responses=responses))
    if mode == "conversation":
        model = model.conversation()
    if mode == "chain":
        replies = [r async for r in model.chain("Hi").responses()]
        response = replies[0]
    else:
        response = await model.prompt("Hi")
    assert await response.text() == "Hello"
    request = json.loads(httpx2_mock.get_requests()[0].content)
    assert request["model"] == "local"
    assert request["stream"] is False


@pytest.mark.asyncio
async def test_installed_model_aliases():
    store_keys(chosen="chosen-key")
    client = make_llm(models=[{"model": "4o", "key": "chosen"}])
    assert [m.model_id for m in await client.models()] == ["gpt-4o"]
    assert (await client.model("4o"))._key == "chosen-key"
    client = make_llm(models=[{"model": "4o", "key": "chosen"}], blocked_models=["4o"])
    assert await client.models() == []
    with pytest.raises(ModelNotFoundError):
        await client.model("4o")


@pytest.mark.asyncio
async def test_listed_responses_model_explicit_key_in_chat_fallback(
    httpx2_mock, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-send")
    store_keys(chosen="stored-key")
    models = await make_llm(models=[endpoint(responses=True, key="chosen")]).models()
    add_response(httpx2_mock, False)
    response = models[0].prompt(
        "Hi", stream=False, chat_completions=True, key="explicit-key"
    )
    assert await response.text() == "Hello"
    assert (
        httpx2_mock.get_requests()[0].headers["authorization"] == "Bearer explicit-key"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("key_name", [None, "chosen"])
@pytest.mark.parametrize("stream", [False, True])
async def test_native_chat_fallback_does_not_reinterpret_literal_keys(
    httpx2_mock, monkeypatch, key_name, stream
):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-send-env")
    store_keys(
        openai="do-not-send-provider",
        DUMMY_KEY="do-not-send-dummy-alias",
        chosen="OPENAI_API_KEY",
        OPENAI_API_KEY="do-not-send-literal-alias",
    )
    ref = endpoint(responses=True, **({"key": key_name} if key_name else {}))
    model = await make_llm(default_model=ref).model()
    add_response(httpx2_mock, False, stream=stream)
    response = await model.prompt("Hi", stream=stream, chat_completions=True)
    assert await response.text() == "Hello"
    expected = "OPENAI_API_KEY" if key_name else "DUMMY_KEY"
    assert (
        httpx2_mock.get_requests()[0].headers["authorization"] == f"Bearer {expected}"
    )


@pytest.mark.asyncio
async def test_responses_reasoning_is_opt_in(httpx2_mock):
    client = make_llm(default_model=endpoint(responses=True, reasoning=True))
    model = await client.model()
    for options in ({}, {"reasoning_summary": "auto"}):
        add_response(httpx2_mock, True)
        response = await model.prompt("Hi", stream=False, **options)
        await response.text()
    first, second = [json.loads(r.content) for r in httpx2_mock.get_requests()]
    assert "reasoning" not in first
    assert second["reasoning"]["summary"] == "auto"


@pytest.mark.asyncio
async def test_responses_server_side_tool(httpx2_mock):
    add_response(httpx2_mock, True)
    model = await make_llm(default_model=endpoint(responses=True)).model()
    spec = {"type": "provider:web_search", "max_results": 2}
    response = await model.prompt(
        "Search", tools=[llm.ServerSideTool(spec)], stream=False
    )
    await response.text()
    assert json.loads(httpx2_mock.get_requests()[0].content)["tools"] == [spec]
