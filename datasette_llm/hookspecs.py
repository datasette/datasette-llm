"""
Plugin hook specifications for datasette-llm.
"""

from pluggy import HookspecMarker

hookspec = HookspecMarker("datasette")


@hookspec
def llm_filter_models(datasette, models, actor, purpose):
    """
    Filter the list of available models.

    Called when listing models via llm.models(). Can be used to restrict
    model access based on actor permissions, purpose, or other criteria.
    Can return a coroutine for async filtering.

    Args:
        datasette: The Datasette instance
        models: List of model objects from llm library
        actor: The current actor (may be None)
        purpose: The purpose string (may be None)

    Returns:
        Filtered list of models, or None to pass through unchanged
    """
    pass


@hookspec
def llm_default_model(datasette, purpose, actor):
    """
    Return the default model ID for a purpose.

    Called when model() or group() is called without an explicit model_id.
    Can return a coroutine for async lookups.

    Args:
        datasette: The Datasette instance
        purpose: The purpose string (may be None)
        actor: The current actor (may be None)

    Returns:
        Model ID string, or None to use configuration defaults
    """
    pass


@hookspec
def llm_prompt_context(datasette, model_id, prompt, purpose):
    """
    Return an async context manager factory to wrap LLM prompt execution.

    This hook allows plugins to intercept and wrap LLM prompts, enabling:
    - Budget/quota checking before prompts execute
    - Token usage tracking after prompts complete
    - Blocking prompts that violate policies
    - Auditing and permissions based on purpose

    Args:
        datasette: The Datasette instance
        model_id: The ID of the model being used (e.g., "gpt-5.4-mini", "echo")
        prompt: The prompt text being sent
        purpose: Optional string identifying what this prompt is for
                 (e.g., "enrichments", "sql-assistant")

    Returns:
        A callable that takes a PromptResult and returns an async context manager,
        or None if this plugin doesn't need to wrap this prompt.

    Note:
        The response may still be streaming when the context manager exits.
        To track usage after completion, register an on_done callback:
            await result.response.on_done(my_callback)

    Example:
        @hookimpl
        def llm_prompt_context(datasette, model_id, prompt, purpose):
            @asynccontextmanager
            async def wrapper(result):
                # Before the prompt executes
                yield
                # After prompt() returns - response may still be streaming
                # Register callback for when response completes
                async def on_complete(response):
                    usage = await response.usage()
                    await record_usage(purpose, usage)
                await result.response.on_done(on_complete)

            return wrapper
    """
    pass


@hookspec
def llm_group_exit(datasette, group):
    """
    Called when a group context manager exits.

    This hook allows plugins to perform cleanup, settlement, or logging
    when a batch of grouped prompts completes.

    Args:
        datasette: The Datasette instance
        group: The Group object containing model_id, purpose, actor, and _responses

    Returns:
        Can return a coroutine which will be awaited.

    Example:
        @hookimpl
        def llm_group_exit(datasette, group):
            async def cleanup():
                await settle_reservation(group)
            return cleanup()
    """
    pass
