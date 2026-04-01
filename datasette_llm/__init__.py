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


def _parse_model_ref(ref) -> tuple:
    """
    Parse a model reference that's either a string or a dict.
    Returns (model_id, key_secret_name_or_none).
    """
    if isinstance(ref, dict):
        return ref["model"], ref.get("key")
    return ref, None


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
def register_secrets(datasette):
    """Register API key secrets for all installed LLM models and config overrides."""
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

    # Also register any custom key names from model-ref dicts in config
    config = datasette.plugin_config("datasette-llm") or {}
    for ref in _iter_model_refs(config):
        _, key_name = _parse_model_ref(ref)
        if key_name and key_name not in seen:
            seen.add(key_name)
            secrets.append(
                Secret(
                    name=key_name,
                    description=f"Custom API key override: {key_name}",
                )
            )

    return secrets


def _iter_model_refs(config):
    """Yield all model references (strings or dicts) from plugin config."""
    default = config.get("default_model")
    if default:
        yield default
    for purpose_config in config.get("purposes", {}).values():
        model = purpose_config.get("model")
        if model:
            yield model
        for entry in purpose_config.get("models", []):
            yield entry


@hookimpl(tryfirst=True)
def llm_filter_models(datasette, models, purpose):
    """Apply config-based model filtering (allowlists and blocklists)."""
    config = datasette.plugin_config("datasette-llm") or {}
    if not config:
        return

    if purpose:
        purpose_config = config.get("purposes", {}).get(purpose, {})
        purpose_allowed_raw = purpose_config.get("models")
        global_allowed = config.get("models")
        if purpose_allowed_raw or global_allowed:
            purpose_allowed_ids = (
                {_parse_model_ref(entry)[0] for entry in purpose_allowed_raw}
                if purpose_allowed_raw
                else set()
            )
            # Auto-include the purpose's default model
            purpose_default = purpose_config.get("model")
            if purpose_default:
                purpose_allowed_ids.add(_parse_model_ref(purpose_default)[0])
            combined = purpose_allowed_ids | set(global_allowed or [])
            models = [m for m in models if m.model_id in combined]
        purpose_blocked = purpose_config.get("blocked_models")
        if purpose_blocked:
            models = [m for m in models if m.model_id not in purpose_blocked]
    else:
        allowed = config.get("models")
        if allowed:
            allowed_ids = set(allowed)
            # Auto-include the global default model
            global_default = config.get("default_model")
            if global_default:
                allowed_ids.add(_parse_model_ref(global_default)[0])
            models = [m for m in models if m.model_id in allowed_ids]

    blocked = config.get("blocked_models")
    if blocked:
        models = [m for m in models if m.model_id not in blocked]

    return models


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
    the responses after the prompt executes. For chain() calls,
    callbacks can be registered before responses are available and
    will be attached as responses are produced.
    """

    response: Optional[llm_library.AsyncResponse] = field(default=None)
    responses: List[llm_library.AsyncResponse] = field(default_factory=list)
    purpose: Optional[str] = field(default=None)
    group: Optional[Group] = field(default=None)
    _response_done_callbacks: list = field(default_factory=list)

    async def add_response(self, response: llm_library.AsyncResponse) -> None:
        """Record a response and attach any registered completion callbacks."""
        if self.response is None:
            self.response = response
        self.responses.append(response)
        for callback in self._response_done_callbacks:
            await response.on_done(callback)

    async def on_response_done(self, callback) -> None:
        """
        Register a callback for when tracked responses complete.

        The callback is attached to existing responses immediately and to any
        future responses added by a chain.
        """
        self._response_done_callbacks.append(callback)
        for response in self.responses:
            await response.on_done(callback)


class WrappedAsyncChainResponse:
    """
    Wraps an AsyncChainResponse so hooks can observe each yielded response.
    """

    def __init__(
        self,
        chain_response,
        context_factories,
        result: PromptResult,
        group: Optional[Group] = None,
    ):
        self._chain_response = chain_response
        self._context_factories = context_factories
        self._result = result
        self._group = group
        self._prepared = False
        self._prepare_lock = asyncio.Lock()

    async def _prepare(self) -> None:
        """
        Run hook enter/exit logic once before the chain starts yielding responses.

        This preserves the existing hook pattern where plugins do setup before
        the call and register response handlers after yield.
        """
        if self._prepared:
            return
        async with self._prepare_lock:
            if self._prepared:
                return
            async with AsyncExitStack() as stack:
                for factory in self._context_factories:
                    if factory is not None:
                        ctx = factory(self._result)
                        await stack.enter_async_context(ctx)
            self._prepared = True

    async def responses(self):
        await self._prepare()
        async for response in self._chain_response.responses():
            await self._result.add_response(response)
            if self._group is not None:
                self._group._responses.append(response)
            yield response

    def __getattr__(self, name):
        return getattr(self._chain_response, name)


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
            await result.add_response(response)

            # Track response in group for forced completion on group exit
            if self._group is not None:
                self._group._responses.append(response)

        return response

    def chain(self, prompt_text: str, **kwargs):
        """
        Execute a conversation chain with tool support.

        Wrap each yielded response in the existing prompt hook lifecycle,
        so hook implementations can track chain responses the same way they
        track direct prompt() responses.
        """
        context_factories = pm.hook.llm_prompt_context(
            datasette=self._datasette,
            model_id=self._model_id,
            prompt=prompt_text,
            purpose=self._purpose,
            actor=self._actor,
        )
        result = PromptResult(purpose=self._purpose, group=self._group)
        chain_response = self._conversation.chain(prompt_text, key=self._key, **kwargs)
        return WrappedAsyncChainResponse(
            chain_response=chain_response,
            context_factories=context_factories,
            result=result,
            group=self._group,
        )


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

    def chain(self, prompt_text: str, **kwargs):
        """
        Execute a chain with tool support via a wrapped conversation.

        Returns an AsyncChainResponse.
        """
        return self.conversation().chain(prompt_text, **kwargs)

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
            await result.add_response(response)

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

    def _get_purpose_key_for_model(
        self, model_id: str, purpose: Optional[str]
    ) -> Optional[str]:
        """
        Look up a key secret name for a model from purpose or default_model config.

        Resolution order:
        1. Purpose's models list (dict entries with matching model)
        2. Purpose's model default (if dict with matching model)
        3. default_model (if dict with matching model)

        Returns the secret name string, or None.
        """
        config = self._get_config()

        if purpose:
            purpose_config = config.get("purposes", {}).get(purpose, {})

            # Check purpose models list first (most specific)
            for entry in purpose_config.get("models", []):
                ref_id, ref_key = _parse_model_ref(entry)
                if ref_id == model_id and ref_key:
                    return ref_key

            # Check purpose default model
            purpose_model = purpose_config.get("model")
            if purpose_model:
                ref_id, ref_key = _parse_model_ref(purpose_model)
                if ref_id == model_id and ref_key:
                    return ref_key

        # Check global default_model
        default = config.get("default_model")
        if default:
            ref_id, ref_key = _parse_model_ref(default)
            if ref_id == model_id and ref_key:
                return ref_key

        return None

    async def get_key_for_model(
        self, model, actor: Optional[dict] = None, purpose: Optional[str] = None
    ) -> Optional[str]:
        """
        Resolve API key for a model.

        Resolution order:
        1. Purpose/config key override (from model-ref dicts)
        2. datasette-secrets (env var DATASETTE_SECRETS_<KEY>_API_KEY or database)
        3. llm's key resolution (keys.json, env vars like OPENAI_API_KEY)

        Args:
            model: The model object (has .needs_key, .key_env_var attributes)
            actor: Optional actor for per-user key resolution
            purpose: Optional purpose for purpose-specific key lookup

        Returns:
            API key string, or None if no key found
        """
        if model.needs_key is None:
            return None

        # 0. Check for purpose/config key override
        key_override = self._get_purpose_key_for_model(model.model_id, purpose)
        if key_override:
            actor_id = actor.get("id") if actor else None
            key = await get_secret(self._datasette, key_override, actor_id)
            if key:
                return key

        # 1. Try datasette-secrets
        secret_name = f"{model.needs_key.upper()}_API_KEY"
        actor_id = actor.get("id") if actor else None
        key = await get_secret(self._datasette, secret_name, actor_id)
        if key:
            return key

        # 2. Fall back to llm's key resolution
        return llm_library.get_key(key_alias=model.needs_key, env_var=model.key_env_var)

    async def model_has_key(
        self, model, actor: Optional[dict] = None, purpose: Optional[str] = None
    ) -> bool:
        """Check if a model has a usable API key configured."""
        if model.needs_key is None:
            return True  # Model doesn't need a key
        key = await self.get_key_for_model(model, actor, purpose=purpose)
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
                    model_id, _ = _parse_model_ref(purpose_model)
                    return model_id

        # Global default
        default = config.get("default_model")
        if default:
            model_id, _ = _parse_model_ref(default)
            return model_id

        raise ModelNotFoundError(
            "No model_id provided and no default_model configured. "
            "Either pass a model_id or configure default_model in datasette-llm settings."
        )

    def _sort_models(self, models: List, config: dict, purpose: Optional[str]) -> List:
        """
        Sort models by priority:
        1. Default model (purpose-specific or global) first
        2. Purpose-specific models list in config order
        3. Global models list in config order
        4. Any remaining models in their original order

        Models already produced by a higher-priority tier are skipped.
        """
        purpose_config = config.get("purposes", {}).get(purpose, {}) if purpose else {}
        purpose_models_list = purpose_config.get("models")
        global_models_list = config.get("models")

        # Resolve default model ID: purpose-specific, then global
        default_model_id = None
        if purpose:
            raw = purpose_config.get("model")
            if raw:
                default_model_id, _ = _parse_model_ref(raw)
        if not default_model_id:
            raw = config.get("default_model")
            if raw:
                default_model_id, _ = _parse_model_ref(raw)

        # Build a lookup from model_id to model object
        model_by_id = {m.model_id: m for m in models}
        available_ids = set(model_by_id.keys())

        result = []
        seen = set()

        def add(model_id):
            if model_id in available_ids and model_id not in seen:
                seen.add(model_id)
                result.append(model_by_id[model_id])

        # 1. Default model first
        if default_model_id:
            add(default_model_id)

        # 2. Purpose-specific models in config order
        if purpose_models_list:
            for entry in purpose_models_list:
                entry_id, _ = _parse_model_ref(entry)
                add(entry_id)

        # 3. Global models in config order
        if global_models_list:
            for model_id in global_models_list:
                add(model_id)

        # 4. Any remaining models in original order
        for m in models:
            if m.model_id not in seen:
                seen.add(m.model_id)
                result.append(m)

        return result

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
        key = await self.get_key_for_model(model, actor, purpose=purpose)

        return WrappedAsyncModel(
            model, self._datasette, purpose=purpose, key=key, actor=actor
        )

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

        # Filter to models with keys (if configured, default True)
        if config.get("require_keys", True):
            # Deduplicate key lookups - many models share the same key
            unique_keys = {
                model.needs_key for model in all_models if model.needs_key is not None
            }
            key_available = {}
            # Pick one representative model per unique key
            key_to_model = {}
            for m in all_models:
                if m.needs_key and m.needs_key not in key_to_model:
                    key_to_model[m.needs_key] = m
            for needs_key in unique_keys:
                key_available[needs_key] = await self.model_has_key(
                    key_to_model[needs_key], actor, purpose=purpose
                )
            all_models = [
                m
                for m in all_models
                if m.needs_key is None or key_available.get(m.needs_key, False)
            ]

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

        # Sort models: default model first, then purpose-specific models
        # in config order, then global models in config order
        all_models = self._sort_models(all_models, config, purpose)

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
        key = await self.get_key_for_model(model, actor, purpose=purpose)

        wrapped = WrappedAsyncModel(
            model,
            self._datasette,
            purpose=purpose,
            group=group_obj,
            key=key,
            actor=actor,
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
