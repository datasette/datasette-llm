from contextlib import asynccontextmanager

import llm
import pytest
from datasette import hookimpl
from datasette.app import Datasette
from datasette.plugins import pm
from datasette_secrets import Secret, get_secret, get_secrets

from datasette_llm import LLM, ModelConfigurationError


def client(**config):
    return LLM(Datasette(memory=True, config={"plugins": {"datasette-llm": config}}))


def endpoint(name, **kwargs):
    return {"model": name, "api_base": "http://localhost:1234/v1", **kwargs}


@pytest.fixture
def large_registry(monkeypatch):
    class SyncModel(llm.KeyModel):
        needs_key = "benchmark"

        def __init__(self, model_id):
            self.model_id = model_id

        def execute(self, prompt, stream, response, conversation, key):
            yield "hello"

    class AsyncModel(llm.AsyncKeyModel):
        needs_key = "benchmark"

        def __init__(self, model_id):
            self.model_id = model_id

        async def execute(self, prompt, stream, response, conversation, key):
            yield "hello"

    monkeypatch.setenv("DATASETTE_SECRETS_BENCHMARK_API_KEY", "benchmark-value")

    @asynccontextmanager
    async def registered(size):
        class Plugin:
            @llm.hookimpl
            def register_models(self, register):
                for i in range(size):
                    name = f"snapshot-{i}"
                    register(SyncModel(name), AsyncModel(name), aliases=[f"alias-{i}"])

        plugin = Plugin()
        llm.plugins.pm.register(plugin)
        try:
            yield
        finally:
            llm.plugins.pm.unregister(plugin)

    return registered


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [30, 300])
@pytest.mark.parametrize("operation", ["model", "models", "group"])
@pytest.mark.parametrize("require_keys", [False, True])
async def test_lookup_registry_passes_are_bounded(
    monkeypatch, large_registry, size, operation, require_keys
):
    async with large_registry(size):
        registry_passes, snapshots = [], []
        original_registry = llm.get_models_with_aliases
        original_snapshot = llm.get_async_model_aliases

        def registry():
            registry_passes.append(True)
            return original_registry()

        def snapshot():
            snapshots.append(True)
            return original_snapshot()

        monkeypatch.setattr(llm, "get_models_with_aliases", registry)
        monkeypatch.setattr(llm, "get_async_model_aliases", snapshot)
        instance = client(
            default_model="alias-0",
            models=[
                {"model": f"alias-{i}", "options": {"temperature": 0.2}}
                for i in range(7)
            ],
            require_keys=require_keys,
        )
        if operation == "models":
            assert len(await instance.models()) == 7
        elif operation == "group":
            async with instance.group() as model:
                assert model.model_id == "snapshot-0"
        else:
            assert (await instance.model("alias-0")).model_id == "snapshot-0"
        assert len(snapshots) == 1
        # Provider secret registration may take a few additional snapshots,
        # but the bound must not grow with registered models or config entries.
        assert len(registry_passes) <= 8


@pytest.mark.asyncio
async def test_lookup_snapshot_is_fresh_for_each_call():
    instance = client(require_keys=False)
    llm.set_alias("chosen", "echo")
    assert (await instance.model("chosen")).model_id == "echo"
    llm.set_alias("chosen", "gpt-4o")
    assert (await instance.model("chosen")).model_id == "gpt-4o"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_ref",
    [
        endpoint("bad", temprature=0.2),
        endpoint("bad", key=None),
        endpoint("bad", options=None),
        {"model": "bad", 0: False, "temprature": 0.2},
    ],
)
async def test_invalid_reference_does_not_break_other_plugins_secrets(
    monkeypatch, invalid_ref
):
    class OtherPlugin:
        @hookimpl
        def register_secrets(self):
            return [Secret("OTHER_KEY")]

    plugin = OtherPlugin()
    pm.register(plugin)
    try:
        monkeypatch.setenv("DATASETTE_SECRETS_OTHER_KEY", "other-value")
        monkeypatch.setenv("DATASETTE_SECRETS_OPENAI_API_KEY", "openai-value")
        instance = client(
            default_model=invalid_ref,
            endpoints=[endpoint("good", key="GOOD_KEY")],
        )
        ds = instance._datasette
        assert await get_secret(ds, "OTHER_KEY") == "other-value"
        assert await get_secret(ds, "OPENAI_API_KEY") == "openai-value"
        assert "GOOD_KEY" in {secret.name for secret in await get_secrets(ds)}
    finally:
        pm.unregister(plugin)


@pytest.mark.asyncio
async def test_colliding_endpoint_does_not_break_native_model_or_apply_its_key(
    monkeypatch,
):
    monkeypatch.setenv("DATASETTE_SECRETS_PRIVATE_KEY", "private-value")
    llm.set_alias("qwen", "echo")
    instance = client(
        endpoints=[endpoint("qwen", key="PRIVATE_KEY", options={"thinking": True})]
    )
    assert "echo" in {model.model_id for model in await instance.models()}
    echo = await instance.model("echo")
    assert echo._key is None
    assert echo._options == {}
    with pytest.raises(ModelConfigurationError, match="qwen.*conflicts"):
        await instance.model("qwen")
    with pytest.raises(ModelConfigurationError, match="qwen.*conflicts"):
        async with instance.group("qwen"):
            pytest.fail("Colliding endpoint selected")


@pytest.mark.asyncio
async def test_purpose_collision_does_not_disable_global_alias():
    llm.set_alias("qwen", "echo")
    instance = client(purposes={"agent": {"endpoints": [endpoint("qwen")]}})
    assert (await instance.model("qwen")).model_id == "echo"
    assert (await instance.model("echo", purpose="agent")).model_id == "echo"
    with pytest.raises(ModelConfigurationError):
        await instance.model("qwen", purpose="agent")


@pytest.mark.asyncio
async def test_endpoint_registry_does_not_enable_allowlisting():
    instance = client(
        default_model="gpt-4o-mini",
        endpoints=[endpoint("first"), endpoint("second")],
        require_keys=False,
    )
    ids = {model.model_id for model in await instance.models()}
    assert {"gpt-4o", "gpt-4o-mini", "echo", "first", "second"} <= ids
    assert (await instance.model("gpt-4o")).model_id == "gpt-4o"
    assert (await instance.model("first")).model_id == "first"
    assert (await instance.model("second")).model_id == "second"


@pytest.mark.asyncio
async def test_endpoint_registry_keys_options_and_allowlist(monkeypatch):
    monkeypatch.setenv("DATASETTE_SECRETS_ENDPOINT_KEY", "endpoint-value")
    instance = client(
        default_model="first",
        endpoints=[
            endpoint("first", key="ENDPOINT_KEY", options={"temperature": 0.2}),
            endpoint("second"),
        ],
        models=["echo"],
    )
    assert [model.model_id for model in await instance.models()] == ["first", "echo"]
    model = await instance.model()
    assert model._key == "endpoint-value"
    assert model._options == {"temperature": 0.2}


@pytest.mark.asyncio
async def test_purpose_endpoint_registry_is_scoped_and_overrides_global_settings(
    monkeypatch,
):
    monkeypatch.setenv("DATASETTE_SECRETS_GLOBAL_KEY", "global-value")
    monkeypatch.setenv("DATASETTE_SECRETS_PURPOSE_KEY", "purpose-value")
    instance = client(
        endpoints=[endpoint("shared", key="GLOBAL_KEY", options={"temperature": 0.1})],
        purposes={
            "agent": {
                "model": "private",
                "endpoints": [
                    endpoint("private"),
                    endpoint("shared", key="PURPOSE_KEY", options={"temperature": 0.2}),
                ],
            }
        },
    )
    assert "private" not in {model.model_id for model in await instance.models()}
    assert "private" in {
        model.model_id for model in await instance.models(purpose="agent")
    }
    assert (await instance.model(purpose="agent")).model_id == "private"
    global_model = await instance.model("shared")
    purpose_model = await instance.model("shared", purpose="agent")
    assert (global_model._key, global_model._options) == (
        "global-value",
        {"temperature": 0.1},
    )
    assert (purpose_model._key, purpose_model._options) == (
        "purpose-value",
        {"temperature": 0.2},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["model", "default", "group"])
async def test_missing_provider_key_has_actionable_error(operation):
    instance = client(default_model="gpt-4o-mini")
    with pytest.raises(
        llm.NeedsKeyException, match="llm keys set openai.*OPENAI_API_KEY"
    ):
        if operation == "group":
            async with instance.group():
                pytest.fail("Missing key accepted")
        else:
            await instance.model("gpt-4o-mini" if operation == "model" else None)
