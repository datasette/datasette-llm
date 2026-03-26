import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

import llm as llm_library
from datasette import hookimpl
from datasette.plugins import pm
from datasette_secrets import Secret, get_secret

from . import hookspecs

# Register our plugin hooks
pm.add_hookspecs(hookspecs)

if TYPE_CHECKING:
    from datasette.app import Datasette


# URLs for obtaining API keys for common providers
KEY_OBTAIN_URLS = {
    "openai": ("https://platform.openai.com/api-keys", "Get an OpenAI API key"),
    "anthropic": (
        "https://console.anthropic.com/settings/keys",
        "Get an Anthropic API key",
    ),
    "google": ("https://aistudio.google.com/app/apikey", "Get a Google AI API key"),
    "groq": ("https://console.groq.com/keys", "Get a Groq API key"),
    "mistral": ("https://console.mistral.ai/api-keys/", "Get a Mistral API key"),
    "cohere": ("https://dashboard.cohere.com/api-keys", "Get a Cohere API key"),
}


@dataclass
class Purpose:
    """
    Describes a purpose that a plugin uses for LLM prompts.

    Purposes help administrators understand what LLM operations are
    available and configure which model should serve each one.

    Attributes:
        name: Unique identifier for this purpose (e.g., "query-assistant", "enrichments")
        description: Human-readable description of what this purpose does
    """

    name: str
    description: Optional[str] = None


def get_purposes(datasette) -> List[Purpose]:
    """
    Collect all registered purposes from plugins.

    Deduplicates by name (first registration wins).

    Returns:
        List of Purpose instances
    """
    purposes = []
    seen = set()
    for result in pm.hook.register_llm_purposes(datasette=datasette):
        for purpose in result:
            if purpose.name in seen:
                continue
            seen.add(purpose.name)
            purposes.append(purpose)
    return purposes


class ModelNotFoundError(Exception):
    """Raised when a requested model is not available."""

    pass


@hookimpl
def register_secrets():
    """Register API key secrets for all installed LLM models."""
    secrets = []
    seen = set()

    for model in llm_library.get_models():
        if model.needs_key and model.needs_key not in seen:
            seen.add(model.needs_key)
            key_name = f"{model.needs_key.upper()}_API_KEY"
            obtain_url, obtain_label = KEY_OBTAIN_URLS.get(
                model.needs_key, (None, None)
            )
            secrets.append(
                Secret(
                    name=key_name,
                    description=f"API key for {model.needs_key} models",
                    obtain_url=obtain_url,
                    obtain_label=obtain_label,
                )
            )

    return secrets


@dataclass
class Group:
    """
    Represents a group of prompts that are logically related.

    Passed to PromptResult so hooks can identify grouped prompts
    and handle them differently (e.g., use shared reservation).
    """

    model_id: str
    purpose: Optional[str] = None
    actor: Optional[dict] = None
    # Responses tracked for forced completion on group exit
    _responses: List[llm_library.AsyncResponse] = field(default_factory=list)


@dataclass
class PromptResult:
    """
    Container for prompt execution results, populated by the wrapper.

    Context managers from hooks receive this object and can access
    the response after the prompt executes.
    """

    response: Optional[llm_library.AsyncResponse] = field(default=None)
    purpose: Optional[str] = field(default=None)
    group: Optional[Group] = field(default=None)


class WrappedConversation:
    """
    Wraps an llm Conversation to invoke hooks around each prompt.
    """

    def __init__(
        self,
        conversation: llm_library.Conversation,
        datasette: "Datasette",
        model_id: str,
        purpose: Optional[str] = None,
        group: Optional[Group] = None,
        key: Optional[str] = None,
        actor: Optional[dict] = None,
    ):
        self._conversation = conversation
        self._datasette = datasette
        self._model_id = model_id
        self._purpose = purpose
        self._group = group
        self._key = key
        self._actor = actor

    async def prompt(self, prompt_text: str, **kwargs) -> llm_library.AsyncResponse:
        """
        Execute a conversation prompt with hook context managers wrapping the call.
        """
        # Collect context manager factories from all registered hooks
        context_factories = pm.hook.llm_prompt_context(
            datasette=self._datasette,
            model_id=self._model_id,
            prompt=prompt_text,
            purpose=self._purpose,
            actor=self._actor,
        )

        # Create result object that will be populated after the prompt
        result = PromptResult(purpose=self._purpose, group=self._group)

        # Enter all context managers, then execute prompt, then exit
        async with AsyncExitStack() as stack:
            for factory in context_factories:
                if factory is not None:
                    ctx = factory(result)
                    await stack.enter_async_context(ctx)

            # Execute the actual conversation prompt, passing the resolved key
            response = await self._conversation.prompt(
                prompt_text, key=self._key, **kwargs
            )

            # Populate result so context managers can access it on exit
            result.response = response

            # Track response in group for forced completion on group exit
            if self._group is not None:
                self._group._responses.append(response)

        return response


class WrappedAsyncModel:
    """
    Wraps an llm AsyncModel to invoke hooks around prompt execution.
    """

    def __init__(
        self,
        model: llm_library.AsyncModel,
        datasette: "Datasette",
        purpose: Optional[str] = None,
        group: Optional[Group] = None,
        key: Optional[str] = None,
        actor: Optional[dict] = None,
    ):
        self._model = model
        self._datasette = datasette
        self._purpose = purpose
        self._group = group
        self._key = key
        self._actor = actor

    @property
    def model_id(self) -> str:
        return self._model.model_id

    @property
    def purpose(self) -> Optional[str]:
        return self._purpose

    def conversation(self):
        """
        Get a wrapped conversation object for multi-turn interactions.

        Conversation prompts are wrapped by hooks just like direct model prompts.
        """
        return WrappedConversation(
            self._model.conversation(),
            self._datasette,
            self._model.model_id,
            purpose=self._purpose,
            group=self._group,
            key=self._key,
            actor=self._actor,
        )

    async def prompt(self, prompt_text: str, **kwargs) -> llm_library.AsyncResponse:
        """
        Execute a prompt with hook context managers wrapping the call.

        Supports both streaming and non-streaming usage:
        - Non-streaming: `text = await response.text()`
        - Streaming: `async for chunk in response:`

        Hooks can register on_done callbacks to track usage after completion.
        """
        # Collect context manager factories from all registered hooks
        context_factories = pm.hook.llm_prompt_context(
            datasette=self._datasette,
            model_id=self._model.model_id,
            prompt=prompt_text,
            purpose=self._purpose,
            actor=self._actor,
        )

        # Create result object that will be populated after the prompt
        result = PromptResult(purpose=self._purpose, group=self._group)

        # Enter all context managers, then execute prompt, then exit
        async with AsyncExitStack() as stack:
            for factory in context_factories:
                if factory is not None:
                    # Each factory takes the result object and returns a context manager
                    ctx = factory(result)
                    await stack.enter_async_context(ctx)

            # Execute the actual prompt, passing the resolved key
            response = await self._model.prompt(prompt_text, key=self._key, **kwargs)

            # Populate result so context managers can access it on exit
            # Note: response may still be streaming - hooks should use
            # response.on_done() to track usage after completion
            result.response = response

            # Track response in group for forced completion on group exit
            if self._group is not None:
                self._group._responses.append(response)

        return response


class LLM:
    """
    Main entry point for using LLM models in Datasette plugins.

    Wraps the llm library to provide hook integration for tracking,
    accounting, and policy enforcement.
    """

    def __init__(self, datasette: "Datasette"):
        self._datasette = datasette

    def _get_config(self) -> dict:
        """Get plugin configuration."""
        return self._datasette.plugin_config("datasette-llm") or {}

    async def get_key_for_model(
        self, model, actor: Optional[dict] = None
    ) -> Optional[str]:
        """
        Resolve API key for a model.

        Resolution order:
        1. datasette-secrets (env var DATASETTE_SECRETS_<KEY>_API_KEY or database)
        2. llm's key resolution (keys.json, env vars like OPENAI_API_KEY)

        Args:
            model: The model object (has .needs_key, .key_env_var attributes)
            actor: Optional actor for per-user key resolution

        Returns:
            API key string, or None if no key found
        """
        if model.needs_key is None:
            return None

        # 1. Try datasette-secrets
        secret_name = f"{model.needs_key.upper()}_API_KEY"
        actor_id = actor.get("id") if actor else None
        key = await get_secret(self._datasette, secret_name, actor_id)
        if key:
            return key

        # 2. Fall back to llm's key resolution
        return llm_library.get_key(key_alias=model.needs_key, env_var=model.key_env_var)

    async def model_has_key(self, model, actor: Optional[dict] = None) -> bool:
        """Check if a model has a usable API key configured."""
        if model.needs_key is None:
            return True  # Model doesn't need a key
        key = await self.get_key_for_model(model, actor)
        return key is not None

    async def _resolve_model_id(
        self,
        model_id: Optional[str],
        purpose: Optional[str],
        actor: Optional[dict],
    ) -> str:
        """
        Resolve the model ID to use.

        Resolution order:
        1. Explicit model_id argument (if provided)
        2. llm_default_model hook
        3. Purpose-specific config: purposes.<purpose>.model
        4. Global default: default_model
        5. Error if none configured
        """
        if model_id is not None:
            return model_id

        # Try hooks first
        for result in pm.hook.llm_default_model(
            datasette=self._datasette,
            purpose=purpose,
            actor=actor,
        ):
            if result is not None:
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None:
                    return result

        # Try configuration
        config = self._get_config()

        # Purpose-specific default
        if purpose:
            purposes = config.get("purposes", {})
            if purpose in purposes:
                purpose_model = purposes[purpose].get("model")
                if purpose_model:
                    return purpose_model

        # Global default
        default = config.get("default_model")
        if default:
            return default

        raise ModelNotFoundError(
            "No model_id provided and no default_model configured. "
            "Either pass a model_id or configure default_model in datasette-llm settings."
        )

    async def model(
        self,
        model_id: Optional[str] = None,
        purpose: Optional[str] = None,
        actor: Optional[dict] = None,
    ) -> "WrappedAsyncModel":
        """
        Get an async model wrapped with hook support.

        Args:
            model_id: The model ID (e.g., "gpt-5.4-mini", "echo"). If not provided,
                      uses the default model from configuration.
            purpose: Optional string identifying what this model will be used for
                     (e.g., "enrichments", "sql-assistant"). Used for auditing,
                     permissions, and default model selection.
            actor: Optional actor dict for per-user key resolution and filtering.

        Returns:
            A WrappedAsyncModel that invokes hooks around prompts
        """
        resolved_model_id = await self._resolve_model_id(model_id, purpose, actor)
        model = llm_library.get_async_model(resolved_model_id)

        # Resolve the API key for this model
        key = await self.get_key_for_model(model, actor)

        return WrappedAsyncModel(model, self._datasette, purpose=purpose, key=key, actor=actor)

    async def models(
        self,
        actor: Optional[dict] = None,
        purpose: Optional[str] = None,
    ) -> List:
        """
        Get available models, filtered by configuration, keys, and hooks.

        Args:
            actor: Optional actor for per-user filtering
            purpose: Optional purpose for context-aware filtering

        Returns:
            List of available model objects
        """
        # Start with all async models from llm library
        all_models = list(llm_library.get_async_models())
        config = self._get_config()

        # Apply static configuration filters
        allowed = config.get("models")
        if allowed:
            all_models = [m for m in all_models if m.model_id in allowed]

        blocked = config.get("blocked_models")
        if blocked:
            all_models = [m for m in all_models if m.model_id not in blocked]

        # Filter to models with keys (if configured, default True)
        if config.get("require_keys", True):
            available = []
            for model in all_models:
                if await self.model_has_key(model, actor):
                    available.append(model)
            all_models = available

        # Apply hook filters (may be async)
        for result in pm.hook.llm_filter_models(
            datasette=self._datasette,
            models=all_models,
            actor=actor,
            purpose=purpose,
        ):
            if result is not None:
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None:
                    all_models = result

        return all_models

    @asynccontextmanager
    async def group(
        self,
        model_id: Optional[str] = None,
        purpose: Optional[str] = None,
        actor: Optional[dict] = None,
    ):
        """
        Context manager for grouping multiple prompts together.

        Use this for batch operations like enrichments where multiple prompts
        are logically related. Benefits:
        - Hooks can handle grouped prompts differently (e.g., shared reservation)
        - All responses are forced to complete on exit (transactional semantics)

        Args:
            model_id: The model ID (e.g., "gpt-5.4-mini", "echo"). If not provided,
                      uses the default model from configuration.
            purpose: String identifying what this group is for (e.g., "enrichments")
            actor: Optional actor dict for per-user key resolution

        Yields:
            A WrappedAsyncModel that tracks all prompts in the group

        Example:
            async with llm.group(purpose="enrichments") as model:
                for row in rows:
                    response = await model.prompt(f"Process: {row}")
                    text = await response.text()
            # All responses guaranteed complete here
        """
        # Resolve model ID
        resolved_model_id = await self._resolve_model_id(model_id, purpose, actor)

        # Create group to track this batch
        group_obj = Group(model_id=resolved_model_id, purpose=purpose, actor=actor)

        # Get the underlying model and resolve its key
        model = llm_library.get_async_model(resolved_model_id)
        key = await self.get_key_for_model(model, actor)

        wrapped = WrappedAsyncModel(
            model, self._datasette, purpose=purpose, group=group_obj, key=key, actor=actor
        )

        try:
            yield wrapped
        finally:
            # Force completion of all responses for transactional semantics
            for response in group_obj._responses:
                try:
                    await response.text()
                except Exception:
                    # Response may have already been consumed or errored
                    pass

            # Notify hooks that the group is exiting (for settlement, cleanup, etc.)
            # Hooks may return coroutines for async cleanup - await them
            results = pm.hook.llm_group_exit(datasette=self._datasette, group=group_obj)
            for result in results:
                if asyncio.iscoroutine(result):
                    await result


__all__ = [
    "LLM",
    "WrappedAsyncModel",
    "WrappedConversation",
    "PromptResult",
    "Group",
    "Purpose",
    "ModelNotFoundError",
    "get_purposes",
]
