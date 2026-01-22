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
        text = await response.text()

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
