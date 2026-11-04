from datasette import hookimpl
from datasette.app import Datasette
from datasette.plugins import pm
from datasette_llm import LLM
import pytest


@pytest.mark.parametrize("async_filter", [False, True])
@pytest.mark.parametrize(
    "config, supported, expected",
    [
        ({"models": ["echo"]}, {"echo", "echo-needs-key"}, ["echo"]),
        (
            {"purposes": {"agent": {"models": ["echo"]}}},
            {"echo", "echo-needs-key"},
            ["echo"],
        ),
        (
            {"blocked_models": ["echo-needs-key"]},
            {"echo", "echo-needs-key"},
            ["echo"],
        ),
        (
            {"purposes": {"agent": {"blocked_models": ["echo-needs-key"]}}},
            {"echo", "echo-needs-key"},
            ["echo"],
        ),
        ({"models": ["echo"]}, {"echo-needs-key"}, []),
        ({"models": ["echo"]}, set(), []),
        ({"models": ["echo"]}, None, ["echo"]),
    ],
)
@pytest.mark.asyncio
async def test_model_filter_preserves_config(config, supported, expected, async_filter):
    class PricingFilter:
        @hookimpl
        def llm_filter_models(self, models):
            # Like an accounting provider: None means it can price every model.
            filtered = (
                models
                if supported is None
                else [model for model in models if model.model_id in supported]
            )
            if async_filter:

                async def inner():
                    return filtered

                return inner()
            return filtered

    plugin = PricingFilter()
    pm.register(plugin)
    try:
        datasette = Datasette(
            memory=True,
            config={"plugins": {"datasette-llm": {"require_keys": False, **config}}},
        )
        models = await LLM(datasette).models(purpose="agent")
        assert [model.model_id for model in models] == expected
    finally:
        pm.unregister(plugin)


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.asyncio
async def test_model_filters_intersect_independent_of_registration_order(reverse_order):
    class FirstFilter:
        @hookimpl
        def llm_filter_models(self, models):
            return [model for model in models if model.model_id != "echo"]

    class SecondFilter:
        @hookimpl
        async def llm_filter_models(self, models):
            return [model for model in models if model.model_id != "echo-needs-key"]

    plugins = [FirstFilter(), SecondFilter()]
    if reverse_order:
        plugins.reverse()
    for plugin in plugins:
        pm.register(plugin)
    try:
        datasette = Datasette(
            memory=True,
            config={
                "plugins": {
                    "datasette-llm": {
                        "require_keys": False,
                        "models": ["echo", "echo-needs-key", "gpt-4o"],
                    }
                }
            },
        )
        models = await LLM(datasette).models()
        assert [model.model_id for model in models] == ["gpt-4o"]
    finally:
        for plugin in plugins:
            pm.unregister(plugin)
