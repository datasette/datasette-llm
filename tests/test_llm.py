from datasette.app import Datasette
from datasette import hookimpl
from datasette.plugins import pm
from contextlib import asynccontextmanager
import pytest


@pytest.mark.asyncio
async def test_plugin_is_installed():
    datasette = Datasette(memory=True)
    response = await datasette.client.get("/-/plugins.json")
    assert response.status_code == 200
    installed_plugins = {p["name"] for p in response.json()}
    assert "datasette-llm" in installed_plugins


@pytest.mark.asyncio
async def test_model():
    """Test that we can get a model and make a prompt."""
    from datasette_llm import LLM

    datasette = Datasette(memory=True)
    llm = LLM(datasette)

    model = await llm.model("echo")
    response = await model.prompt("Hello, world!")
    text = await response.text()

    # llm-echo returns JSON with the prompt
    assert "Hello, world!" in text


@pytest.mark.asyncio
async def test_model_with_purpose():
    """Test that purpose is passed through to the model."""
    from datasette_llm import LLM

    datasette = Datasette(memory=True)
    llm = LLM(datasette)

    model = await llm.model("echo", purpose="sql-assistant")
    assert model.purpose == "sql-assistant"


@pytest.mark.asyncio
async def test_llm_prompt_context_hook():
    """Test that the llm_prompt_context hook is called and can access the response."""
    from datasette_llm import LLM

    # Track what the hook sees
    hook_calls = []

    class TestPlugin:
        __name__ = "test_plugin"

        @hookimpl
        def llm_prompt_context(self, datasette, model_id, prompt, purpose):
            @asynccontextmanager
            async def tracker(result):
                # Before the prompt
                hook_calls.append(
                    {
                        "phase": "before",
                        "model_id": model_id,
                        "prompt": prompt,
                        "purpose": purpose,
                    }
                )
                yield

                # After the prompt - register on_done for usage tracking
                async def on_complete(response):
                    hook_calls.append(
                        {
                            "phase": "on_done",
                            "model_id": model_id,
                            "response_text": await response.text(),
                        }
                    )

                await result.response.on_done(on_complete)

            return tracker

    plugin = TestPlugin()
    pm.register(plugin)
    try:
        datasette = Datasette(memory=True)
        llm = LLM(datasette)
        model = await llm.model("echo", purpose="test-purpose")

        response = await model.prompt("Test prompt")
        await response.text()

        # Verify hook was called
        assert len(hook_calls) == 2
        assert hook_calls[0]["phase"] == "before"
        assert hook_calls[0]["model_id"] == "echo"
        assert hook_calls[0]["prompt"] == "Test prompt"
        assert hook_calls[0]["purpose"] == "test-purpose"
        assert hook_calls[1]["phase"] == "on_done"
        assert "Test prompt" in hook_calls[1]["response_text"]
    finally:
        pm.unregister(plugin)


@pytest.mark.asyncio
async def test_multiple_context_hooks():
    """Test that multiple hooks are all called in order."""
    from datasette_llm import LLM

    call_order = []

    class Plugin1:
        __name__ = "plugin1"

        @hookimpl
        def llm_prompt_context(self, datasette, model_id, prompt, purpose):
            @asynccontextmanager
            async def tracker(result):
                call_order.append("plugin1_enter")
                yield
                call_order.append("plugin1_exit")

            return tracker

    class Plugin2:
        __name__ = "plugin2"

        @hookimpl
        def llm_prompt_context(self, datasette, model_id, prompt, purpose):
            @asynccontextmanager
            async def tracker(result):
                call_order.append("plugin2_enter")
                yield
                call_order.append("plugin2_exit")

            return tracker

    plugin1 = Plugin1()
    plugin2 = Plugin2()
    pm.register(plugin1)
    pm.register(plugin2)
    try:
        datasette = Datasette(memory=True)
        llm = LLM(datasette)
        model = await llm.model("echo")

        response = await model.prompt("Test")
        await response.text()

        # Both hooks should be called
        assert "plugin1_enter" in call_order
        assert "plugin1_exit" in call_order
        assert "plugin2_enter" in call_order
        assert "plugin2_exit" in call_order
    finally:
        pm.unregister(plugin1)
        pm.unregister(plugin2)


@pytest.mark.asyncio
async def test_hook_can_block_prompt():
    """Test that a hook can raise an exception to block the prompt."""
    from datasette_llm import LLM

    class BlockingPlugin:
        __name__ = "blocking_plugin"

        @hookimpl
        def llm_prompt_context(self, datasette, model_id, prompt, purpose):
            @asynccontextmanager
            async def blocker(result):
                raise PermissionError("Prompt blocked by policy")
                yield  # Never reached

            return blocker

    plugin = BlockingPlugin()
    pm.register(plugin)
    try:
        datasette = Datasette(memory=True)
        llm = LLM(datasette)
        model = await llm.model("echo")

        with pytest.raises(PermissionError, match="Prompt blocked by policy"):
            await model.prompt("Should be blocked")
    finally:
        pm.unregister(plugin)


@pytest.mark.asyncio
async def test_streaming_response():
    """Test that streaming works correctly (response not forced to complete)."""
    from datasette_llm import LLM

    datasette = Datasette(memory=True)
    llm = LLM(datasette)
    model = await llm.model("echo")

    response = await model.prompt("Stream test")

    # Should be able to iterate over chunks
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "Stream test" in full_text


@pytest.mark.asyncio
async def test_group_basic():
    """Test basic group functionality."""
    from datasette_llm import LLM

    datasette = Datasette(memory=True)
    llm = LLM(datasette)

    results = []
    async with llm.group("echo", purpose="test-batch") as model:
        for i in range(3):
            response = await model.prompt(f"Item {i}")
            text = await response.text()
            results.append(text)

    assert len(results) == 3
    assert "Item 0" in results[0]
    assert "Item 1" in results[1]
    assert "Item 2" in results[2]


@pytest.mark.asyncio
async def test_group_forces_completion():
    """Test that group forces completion of unconsumed responses on exit."""
    from datasette_llm import LLM

    datasette = Datasette(memory=True)
    llm = LLM(datasette)

    responses = []
    async with llm.group("echo", purpose="test-batch") as model:
        # Create responses but don't consume them
        for i in range(3):
            response = await model.prompt(f"Item {i}")
            responses.append(response)
        # Exit without calling .text() on any response

    # After exiting group, all responses should be complete
    # (we can verify by checking that .text() returns immediately)
    for i, response in enumerate(responses):
        text = await response.text()
        assert f"Item {i}" in text


@pytest.mark.asyncio
async def test_group_passes_to_hooks():
    """Test that group info is passed to hooks via PromptResult."""
    from datasette_llm import LLM

    hook_data = []

    class GroupTrackingPlugin:
        __name__ = "group_tracking_plugin"

        @hookimpl
        def llm_prompt_context(self, datasette, model_id, prompt, purpose):
            @asynccontextmanager
            async def tracker(result):
                yield
                hook_data.append(
                    {
                        "purpose": result.purpose,
                        "has_group": result.group is not None,
                        "group_purpose": result.group.purpose if result.group else None,
                    }
                )

            return tracker

    plugin = GroupTrackingPlugin()
    pm.register(plugin)
    try:
        datasette = Datasette(memory=True)
        llm = LLM(datasette)

        async with llm.group("echo", purpose="enrichments") as model:
            response = await model.prompt("Test")
            await response.text()

        assert len(hook_data) == 1
        assert hook_data[0]["purpose"] == "enrichments"
        assert hook_data[0]["has_group"] is True
        assert hook_data[0]["group_purpose"] == "enrichments"
    finally:
        pm.unregister(plugin)


# New tests for model filtering and defaults


@pytest.mark.asyncio
async def test_models_returns_available_models():
    """Test that models() returns available models."""
    from datasette_llm import LLM

    # Configure to not require keys (since echo doesn't need one)
    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                }
            }
        },
    )
    llm = LLM(datasette)

    models = await llm.models()
    model_ids = [m.model_id for m in models]

    # Should include the echo model at least
    assert "echo" in model_ids


@pytest.mark.asyncio
async def test_models_filtered_by_config():
    """Test that models can be filtered by configuration."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "models": ["echo"],  # Only allow echo
                }
            }
        },
    )
    llm = LLM(datasette)

    models = await llm.models()
    model_ids = [m.model_id for m in models]

    assert model_ids == ["echo"]


@pytest.mark.asyncio
async def test_models_blocked_by_config():
    """Test that models can be blocked by configuration."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "blocked_models": ["echo"],
                }
            }
        },
    )
    llm = LLM(datasette)

    models = await llm.models()
    model_ids = [m.model_id for m in models]

    assert "echo" not in model_ids


@pytest.mark.asyncio
async def test_default_model_from_config():
    """Test that default_model configuration works."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "default_model": "echo",
                }
            }
        },
    )
    llm = LLM(datasette)

    # Call model() without model_id
    model = await llm.model()
    assert model.model_id == "echo"


@pytest.mark.asyncio
async def test_purpose_specific_model():
    """Test that purpose-specific model configuration works."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "default_model": "echo",
                    "purposes": {
                        "enrichments": {"model": "echo"},
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    # Call model() with purpose
    model = await llm.model(purpose="enrichments")
    assert model.model_id == "echo"


@pytest.mark.asyncio
async def test_model_not_found_error():
    """Test that ModelNotFoundError is raised when no default is configured."""
    from datasette_llm import LLM, ModelNotFoundError

    datasette = Datasette(memory=True)
    llm = LLM(datasette)

    with pytest.raises(ModelNotFoundError):
        await llm.model()  # No model_id, no default configured


@pytest.mark.asyncio
async def test_group_without_model_id():
    """Test that group() works without explicit model_id when default is configured."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "purposes": {
                        "enrichments": {"model": "echo"},
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    async with llm.group(purpose="enrichments") as model:
        assert model.model_id == "echo"
        response = await model.prompt("Test")
        text = await response.text()
        assert "Test" in text


@pytest.mark.asyncio
async def test_llm_filter_models_hook():
    """Test that llm_filter_models hook can filter models."""
    from datasette_llm import LLM

    class FilterPlugin:
        __name__ = "filter_plugin"

        @hookimpl
        def llm_filter_models(self, datasette, models, actor, purpose):
            # Only return models with "echo" in the name
            return [m for m in models if "echo" in m.model_id]

    plugin = FilterPlugin()
    pm.register(plugin)
    try:
        datasette = Datasette(
            memory=True,
            metadata={
                "plugins": {
                    "datasette-llm": {
                        "require_keys": False,
                    }
                }
            },
        )
        llm = LLM(datasette)

        models = await llm.models()
        model_ids = [m.model_id for m in models]

        # Should only have echo
        assert all("echo" in mid for mid in model_ids)
    finally:
        pm.unregister(plugin)


@pytest.mark.asyncio
async def test_llm_default_model_hook():
    """Test that llm_default_model hook can provide default model."""
    from datasette_llm import LLM

    class DefaultModelPlugin:
        __name__ = "default_model_plugin"

        @hookimpl
        def llm_default_model(self, datasette, purpose, actor):
            if purpose == "test-purpose":
                return "echo"
            return None

    plugin = DefaultModelPlugin()
    pm.register(plugin)
    try:
        datasette = Datasette(memory=True)
        llm = LLM(datasette)

        # Should use hook's default for this purpose
        model = await llm.model(purpose="test-purpose")
        assert model.model_id == "echo"
    finally:
        pm.unregister(plugin)


def test_register_llm_purposes():
    """Test that plugins can register purposes via the hook."""
    from datasette_llm import Purpose, get_purposes

    class PurposePlugin:
        __name__ = "purpose_plugin"

        @hookimpl
        def register_llm_purposes(self, datasette):
            return [
                Purpose(
                    name="query-assistant",
                    description="Assists users with writing SQL queries",
                ),
                Purpose(
                    name="enrichments",
                    description="Enriches data with LLM-generated content",
                ),
            ]

    plugin = PurposePlugin()
    pm.register(plugin)
    try:
        datasette = Datasette(memory=True)
        purposes = get_purposes(datasette)

        assert len(purposes) == 2
        assert purposes[0].name == "query-assistant"
        assert purposes[0].description == "Assists users with writing SQL queries"
        assert purposes[1].name == "enrichments"
        assert purposes[1].description == "Enriches data with LLM-generated content"
    finally:
        pm.unregister(plugin)


def test_register_llm_purposes_deduplication():
    """Test that duplicate purpose names are deduplicated (first wins)."""
    from datasette_llm import Purpose, get_purposes

    class Plugin1:
        __name__ = "purpose_plugin1"

        @hookimpl
        def register_llm_purposes(self, datasette):
            return [
                Purpose(name="enrichments", description="First plugin's enrichments"),
            ]

    class Plugin2:
        __name__ = "purpose_plugin2"

        @hookimpl
        def register_llm_purposes(self, datasette):
            return [
                Purpose(name="enrichments", description="Second plugin's enrichments"),
            ]

    plugin1 = Plugin1()
    plugin2 = Plugin2()
    pm.register(plugin1)
    pm.register(plugin2)
    try:
        datasette = Datasette(memory=True)
        purposes = get_purposes(datasette)

        enrichments = [p for p in purposes if p.name == "enrichments"]
        assert len(enrichments) == 1
    finally:
        pm.unregister(plugin1)
        pm.unregister(plugin2)


def test_register_llm_purposes_no_plugins():
    """Test that get_purposes returns empty list when no plugins register purposes."""
    from datasette_llm import get_purposes

    datasette = Datasette(memory=True)
    purposes = get_purposes(datasette)
    assert purposes == []


@pytest.mark.asyncio
async def test_llm_default_model_hook_async():
    """Test that llm_default_model hook can be async."""
    from datasette_llm import LLM

    class AsyncDefaultModelPlugin:
        __name__ = "async_default_model_plugin"

        @hookimpl
        def llm_default_model(self, datasette, purpose, actor):
            async def get_default():
                # Simulate async lookup
                return "echo"

            return get_default()

    plugin = AsyncDefaultModelPlugin()
    pm.register(plugin)
    try:
        datasette = Datasette(memory=True)
        llm = LLM(datasette)

        model = await llm.model(purpose="any")
        assert model.model_id == "echo"
    finally:
        pm.unregister(plugin)


@pytest.mark.asyncio
async def test_purpose_specific_models_filter():
    """Test that purposes can define an allowed models list."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "purposes": {
                        "extract": {
                            "models": ["echo"],
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    # Without purpose, should return all models
    all_models = await llm.models()
    all_ids = [m.model_id for m in all_models]
    assert len(all_ids) > 1

    # With purpose, should only return echo
    filtered = await llm.models(purpose="extract")
    filtered_ids = [m.model_id for m in filtered]
    assert filtered_ids == ["echo"]


@pytest.mark.asyncio
async def test_purpose_specific_blocked_models():
    """Test that purposes can define a blocked models list."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "purposes": {
                        "extract": {
                            "blocked_models": ["echo"],
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    # Without purpose, echo should be present
    all_models = await llm.models()
    assert "echo" in [m.model_id for m in all_models]

    # With purpose, echo should be blocked
    filtered = await llm.models(purpose="extract")
    assert "echo" not in [m.model_id for m in filtered]


@pytest.mark.asyncio
async def test_purpose_models_overrides_global_models():
    """Test that a purpose allowlist can include models not in the global allowlist."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "models": ["gpt-4o"],  # Global: only gpt-4o
                    "purposes": {
                        "extract": {
                            "models": ["echo"],  # Purpose: only echo
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    # Without purpose, only gpt-4o available
    global_models = await llm.models()
    global_ids = [m.model_id for m in global_models]
    assert "gpt-4o" in global_ids
    assert "echo" not in global_ids

    # With purpose, echo is available even though it's not in global list
    purpose_models = await llm.models(purpose="extract")
    purpose_ids = [m.model_id for m in purpose_models]
    assert "echo" in purpose_ids


@pytest.mark.asyncio
async def test_purpose_blocked_models_overrides_global():
    """Test that a purpose can block a model that is globally allowed."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "models": ["echo", "gpt-4o"],  # Both globally allowed
                    "purposes": {
                        "extract": {
                            "blocked_models": ["echo"],  # Block echo for this purpose
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    # Without purpose, echo is available
    global_models = await llm.models()
    global_ids = [m.model_id for m in global_models]
    assert "echo" in global_ids

    # With purpose, echo is blocked
    purpose_models = await llm.models(purpose="extract")
    purpose_ids = [m.model_id for m in purpose_models]
    assert "echo" not in purpose_ids
    assert "gpt-4o" in purpose_ids


# Ordering tests


@pytest.mark.asyncio
async def test_global_models_order_matches_config():
    """Models should be returned in the order listed in the global models config."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "models": ["gpt-4o-mini", "echo", "gpt-4o"],
                }
            }
        },
    )
    llm = LLM(datasette)

    model_ids = [m.model_id for m in await llm.models()]
    assert model_ids == ["gpt-4o-mini", "echo", "gpt-4o"]


@pytest.mark.asyncio
async def test_default_model_is_first():
    """The default model should be promoted to the top of the list."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "models": ["gpt-4o-mini", "echo", "gpt-4o"],
                    "default_model": "gpt-4o",
                }
            }
        },
    )
    llm = LLM(datasette)

    model_ids = [m.model_id for m in await llm.models()]
    # gpt-4o promoted to first, rest in config order
    assert model_ids == ["gpt-4o", "gpt-4o-mini", "echo"]


@pytest.mark.asyncio
async def test_purpose_default_model_is_first():
    """The purpose-specific default model should be promoted to the top."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "models": ["gpt-4o-mini", "echo", "gpt-4o"],
                    "purposes": {
                        "extract": {
                            "model": "echo",
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    model_ids = [m.model_id for m in await llm.models(purpose="extract")]
    # echo promoted to first, rest in global config order
    assert model_ids == ["echo", "gpt-4o-mini", "gpt-4o"]


@pytest.mark.asyncio
async def test_purpose_models_order_then_global_order():
    """
    Purpose models come first in their config order,
    then remaining global models in their config order.
    """
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "models": [
                        "gpt-4o",
                        "gpt-4o-mini",
                        "echo",
                        "gpt-4.1",
                        "gpt-4.1-mini",
                    ],
                    "purposes": {
                        "extract": {
                            "models": ["echo", "gpt-4.1-mini"],
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    model_ids = [m.model_id for m in await llm.models(purpose="extract")]
    # Purpose models first (echo, gpt-4.1-mini), then remaining global
    # models in their global order (gpt-4o, gpt-4o-mini, gpt-4.1)
    assert model_ids == ["echo", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "gpt-4.1"]


@pytest.mark.asyncio
async def test_purpose_models_order_with_default():
    """
    Default model first, then remaining purpose models in config order,
    then remaining global models in global config order.
    """
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "models": [
                        "gpt-4o",
                        "gpt-4o-mini",
                        "echo",
                        "gpt-4.1",
                        "gpt-4.1-mini",
                    ],
                    "purposes": {
                        "extract": {
                            "model": "gpt-4.1-mini",
                            "models": ["echo", "gpt-4.1-mini"],
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    model_ids = [m.model_id for m in await llm.models(purpose="extract")]
    # Default (gpt-4.1-mini) first, then remaining purpose models (echo),
    # then remaining global models in global order (gpt-4o, gpt-4o-mini, gpt-4.1)
    assert model_ids == ["gpt-4.1-mini", "echo", "gpt-4o", "gpt-4o-mini", "gpt-4.1"]


@pytest.mark.asyncio
async def test_purpose_models_order_with_blocked():
    """
    Blocked models are removed from the final ordered list.
    """
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "models": [
                        "gpt-4o",
                        "gpt-4o-mini",
                        "echo",
                        "gpt-4.1",
                        "gpt-4.1-mini",
                    ],
                    "purposes": {
                        "extract": {
                            "model": "gpt-4.1-mini",
                            "models": ["echo", "gpt-4.1-mini"],
                            "blocked_models": ["gpt-4o"],
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    model_ids = [m.model_id for m in await llm.models(purpose="extract")]
    # Same as above but gpt-4o is blocked
    assert model_ids == ["gpt-4.1-mini", "echo", "gpt-4o-mini", "gpt-4.1"]


@pytest.mark.asyncio
async def test_global_default_used_when_no_purpose_default():
    """
    When a purpose has no default model, the global default is promoted to top.
    """
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "default_model": "echo",
                    "models": ["gpt-4o", "gpt-4o-mini", "echo"],
                }
            }
        },
    )
    llm = LLM(datasette)

    # Even with a purpose that has no specific default, global default goes first
    model_ids = [m.model_id for m in await llm.models(purpose="anything")]
    assert model_ids == ["echo", "gpt-4o", "gpt-4o-mini"]


# Model reference with key tests


@pytest.mark.asyncio
async def test_purpose_model_dict_with_key(monkeypatch):
    """Test that purpose model can be a dict with model and key, and the key is used."""
    from datasette_llm import LLM
    import json

    monkeypatch.setenv("DATASETTE_SECRETS_QUERY_ASSISTANT_KEY", "sk-qa-test-1234")
    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "purposes": {
                        "query-assistant": {
                            "model": {
                                "model": "echo-needs-key",
                                "key": "QUERY_ASSISTANT_KEY",
                            },
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    model = await llm.model(purpose="query-assistant")
    assert model.model_id == "echo-needs-key"
    assert model._key == "sk-qa-test-1234"

    # Verify the key is actually passed through to the model
    response = await model.prompt("Hello")
    data = json.loads(await response.text())
    assert data["key"] == "sk-qa-test-1234"


@pytest.mark.asyncio
async def test_purpose_models_list_dict_with_key(monkeypatch):
    """Test that purpose models list entries can be dicts with model and key."""
    from datasette_llm import LLM
    import json

    monkeypatch.setenv("DATASETTE_SECRETS_ENRICHMENTS_KEY", "sk-enrich-test-5678")
    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "purposes": {
                        "enrichments": {
                            "model": "echo",
                            "models": [
                                {"model": "echo-needs-key", "key": "ENRICHMENTS_KEY"},
                                "echo",
                            ],
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    model = await llm.model("echo-needs-key", purpose="enrichments")
    assert model._key == "sk-enrich-test-5678"

    response = await model.prompt("Hello")
    data = json.loads(await response.text())
    assert data["key"] == "sk-enrich-test-5678"


@pytest.mark.asyncio
async def test_default_model_dict_with_key(monkeypatch):
    """Test that default_model can be a dict with model and key fields."""
    from datasette_llm import LLM
    import json

    monkeypatch.setenv("DATASETTE_SECRETS_DEFAULT_CUSTOM_KEY", "sk-default-test-abcd")
    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "default_model": {
                        "model": "echo-needs-key",
                        "key": "DEFAULT_CUSTOM_KEY",
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    model = await llm.model()
    assert model.model_id == "echo-needs-key"
    assert model._key == "sk-default-test-abcd"

    response = await model.prompt("Hello")
    data = json.loads(await response.text())
    assert data["key"] == "sk-default-test-abcd"


@pytest.mark.asyncio
async def test_purpose_key_overrides_default_key(monkeypatch):
    """Test that purpose-specific key takes priority over default_model key."""
    from datasette_llm import LLM
    import json

    monkeypatch.setenv("DATASETTE_SECRETS_DEFAULT_KEY", "sk-default-0000")
    monkeypatch.setenv("DATASETTE_SECRETS_ENRICHMENTS_KEY", "sk-enrichments-1111")
    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "default_model": {
                        "model": "echo-needs-key",
                        "key": "DEFAULT_KEY",
                    },
                    "purposes": {
                        "enrichments": {
                            "model": {
                                "model": "echo-needs-key",
                                "key": "ENRICHMENTS_KEY",
                            },
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    # Purpose-specific key wins
    model = await llm.model(purpose="enrichments")
    response = await model.prompt("Hello")
    data = json.loads(await response.text())
    assert data["key"] == "sk-enrichments-1111"

    # Without purpose, default_model key is used
    model2 = await llm.model()
    response2 = await model2.prompt("Hello")
    data2 = json.loads(await response2.text())
    assert data2["key"] == "sk-default-0000"


@pytest.mark.asyncio
async def test_purpose_models_list_key_takes_priority_over_purpose_model_key(
    monkeypatch,
):
    """Test that a key in the models list overrides the key in the purpose model default."""
    from datasette_llm import LLM
    import json

    monkeypatch.setenv("DATASETTE_SECRETS_DEFAULT_ENRICHMENTS_KEY", "sk-default-enrich")
    monkeypatch.setenv("DATASETTE_SECRETS_SPECIFIC_KEY", "sk-specific-enrich")
    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "purposes": {
                        "enrichments": {
                            "model": {
                                "model": "echo-needs-key",
                                "key": "DEFAULT_ENRICHMENTS_KEY",
                            },
                            "models": [
                                {"model": "echo-needs-key", "key": "SPECIFIC_KEY"},
                            ],
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    # models list entry should win over purpose default model
    model = await llm.model(purpose="enrichments")
    response = await model.prompt("Hello")
    data = json.loads(await response.text())
    assert data["key"] == "sk-specific-enrich"


@pytest.mark.asyncio
async def test_same_provider_different_keys_in_models_list(monkeypatch):
    """Test two models from the same provider can have different keys."""
    from datasette_llm import LLM

    monkeypatch.setenv("DATASETTE_SECRETS_KEY_A", "sk-key-a")
    monkeypatch.setenv("DATASETTE_SECRETS_KEY_B", "sk-key-b")
    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "purposes": {
                        "enrichments": {
                            "model": "echo",
                            "models": [
                                {"model": "gpt-4o", "key": "KEY_A"},
                                {"model": "gpt-4o-mini", "key": "KEY_B"},
                                "echo",
                            ],
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    assert llm._get_purpose_key_for_model("gpt-4o", "enrichments") == "KEY_A"
    assert llm._get_purpose_key_for_model("gpt-4o-mini", "enrichments") == "KEY_B"
    assert llm._get_purpose_key_for_model("echo", "enrichments") is None


@pytest.mark.asyncio
async def test_model_dict_no_key_field():
    """Test that a model dict without key field works (just model name)."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "default_model": {
                        "model": "echo",
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    model = await llm.model()
    assert model.model_id == "echo"
    assert llm._get_purpose_key_for_model("echo", None) is None


@pytest.mark.asyncio
async def test_sort_models_with_dict_refs():
    """Test that model sorting works when purpose config uses dict model refs."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "models": ["gpt-4o-mini", "echo", "gpt-4o"],
                    "purposes": {
                        "extract": {
                            "model": {
                                "model": "echo",
                                "key": "EXTRACT_KEY",
                            },
                            "models": [
                                {"model": "echo", "key": "EXTRACT_KEY"},
                                "gpt-4o",
                            ],
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    model_ids = [m.model_id for m in await llm.models(purpose="extract")]
    # echo promoted to first (purpose default), then gpt-4o (purpose models),
    # then gpt-4o-mini (remaining global)
    assert model_ids == ["echo", "gpt-4o", "gpt-4o-mini"]


@pytest.mark.asyncio
async def test_filter_models_with_dict_refs():
    """Test that model filtering works when purpose models list has dict entries."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "require_keys": False,
                    "purposes": {
                        "extract": {
                            "models": [
                                {"model": "echo", "key": "EXTRACT_KEY"},
                            ],
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    # With purpose, only echo should be available
    filtered = await llm.models(purpose="extract")
    filtered_ids = [m.model_id for m in filtered]
    assert filtered_ids == ["echo"]


@pytest.mark.asyncio
async def test_resolve_model_id_with_dict_default():
    """Test that _resolve_model_id handles dict default_model."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "default_model": {
                        "model": "echo-needs-key",
                        "key": "CUSTOM_KEY",
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    model_id = await llm._resolve_model_id(None, None, None)
    assert model_id == "echo-needs-key"


@pytest.mark.asyncio
async def test_resolve_model_id_with_dict_purpose_model():
    """Test that _resolve_model_id handles dict purpose model."""
    from datasette_llm import LLM

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "purposes": {
                        "enrichments": {
                            "model": {
                                "model": "echo-needs-key",
                                "key": "ENRICHMENTS_KEY",
                            },
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    model_id = await llm._resolve_model_id(None, "enrichments", None)
    assert model_id == "echo-needs-key"


@pytest.mark.asyncio
async def test_group_uses_purpose_key(monkeypatch):
    """Test that group() resolves purpose-specific keys correctly."""
    from datasette_llm import LLM
    import json

    monkeypatch.setenv("DATASETTE_SECRETS_GROUP_KEY", "sk-group-test-9999")
    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "purposes": {
                        "enrichments": {
                            "model": {
                                "model": "echo-needs-key",
                                "key": "GROUP_KEY",
                            },
                        },
                    },
                }
            }
        },
    )
    llm = LLM(datasette)

    async with llm.group(purpose="enrichments") as model:
        assert model._key == "sk-group-test-9999"
        response = await model.prompt("Test")
        data = json.loads(await response.text())
        assert data["key"] == "sk-group-test-9999"


@pytest.mark.asyncio
async def test_custom_key_secrets_auto_registered():
    """Test that custom key names in config are auto-registered as datasette-secrets."""

    datasette = Datasette(
        memory=True,
        metadata={
            "plugins": {
                "datasette-llm": {
                    "default_model": {
                        "model": "echo-needs-key",
                        "key": "MY_CUSTOM_KEY",
                    },
                    "purposes": {
                        "enrichments": {
                            "model": {
                                "model": "echo-needs-key",
                                "key": "ENRICHMENTS_CUSTOM_KEY",
                            },
                            "models": [
                                {"model": "gpt-4o", "key": "ENRICHMENTS_GPT4O_KEY"},
                                "echo",
                            ],
                        },
                    },
                }
            }
        },
    )

    from datasette_secrets import get_secrets

    secrets = await get_secrets(datasette)
    secret_names = {s.name for s in secrets}

    assert "MY_CUSTOM_KEY" in secret_names
    assert "ENRICHMENTS_CUSTOM_KEY" in secret_names
    assert "ENRICHMENTS_GPT4O_KEY" in secret_names
