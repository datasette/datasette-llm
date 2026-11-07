import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

import llm as llm_library
from datasette import hookimpl
from datasette.plugins import pm
from datasette_secrets import Secret, get_secret

from . import hookspecs
from .model_refs import ModelConfigurationError, ModelReference
from .model_lookup import (
    ModelLookup,
    current_lookup,
    iter_model_refs as _iter_model_refs,
)

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
def register_secrets(datasette):
    """Register API key secrets for all installed LLM models and config overrides."""
    secrets = []
    seen = set()

    for model in llm_library.get_models():
        if model.needs_key:
            key_name = f"{model.needs_key.upper()}_API_KEY"
            if key_name in seen:
                continue
            seen.add(key_name)
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

    for key_name in getattr(datasette, "_llm_inline_key_names", ()):
        if key_name not in seen:
            seen.add(key_name)
            secrets.append(
                Secret(name=key_name, description=f"Custom API key: {key_name}")
            )

    # Also register any custom key names from model-ref dicts in config
    config = datasette.plugin_config("datasette-llm") or {}
    for ref in _iter_model_refs(config):
        try:
            key_name = ModelReference.parse(ref).key
        except ModelConfigurationError:
            # A typo in this plugin must not disable every plugin's secrets.
            continue
        if key_name and key_name not in seen:
            seen.add(key_name)
            secrets.append(
                Secret(
                    name=key_name,
                    description=f"Custom API key override: {key_name}",
                )
            )

    return secrets


@hookimpl(tryfirst=True)
def llm_filter_models(datasette, models, purpose):
    """Apply config filtering using the current operation's registry snapshot."""
    active = current_lookup.get()
    if active is not None and active[0] is datasette and active[1].purpose == purpose:
        lookup = active[1]
    else:
        lookup = ModelLookup(datasette.plugin_config("datasette-llm") or {}, purpose)
    return lookup.filter_config(models)


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
    is_chain: bool = field(default=False)
    chain_complete: bool = field(default=False)
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
        self._result.is_chain = True
        self._group = group
        self._prepared = False
        self._prepare_lock = asyncio.Lock()

    async def responses(self):
        async with self._prepare_lock:
            if self._prepared:
                raise RuntimeError("Chain responses can only be iterated once")
            self._prepared = True

        async with AsyncExitStack() as stack:
            for factory in self._context_factories:
                if factory is not None:
                    ctx = factory(self._result)
                    await stack.enter_async_context(ctx)
            try:
                async for response in self._chain_response.responses():
                    await self._result.add_response(response)
                    if self._group is not None:
                        self._group._responses.append(response)
                    yield response
            finally:
                self._result.chain_complete = True

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
        options: Optional[dict] = None,
    ):
        self._conversation = conversation
        self._datasette = datasette
        self._model_id = model_id
        self._purpose = purpose
        self._group = group
        self._key = key
        self._actor = actor
        self._options = options or {}

    def _merged_prompt_kwargs(self, kwargs: dict) -> dict:
        """Merge configured default options into prompt() kwargs.

        prompt() collects options via **kwargs, so we splice options
        directly into the kwargs dict. Caller-supplied kwargs win over
        configured defaults.
        """
        if not self._options:
            return kwargs
        return {**self._options, **kwargs}

    def _merged_chain_kwargs(self, kwargs: dict) -> dict:
        """Merge configured default options into chain() kwargs.

        chain() takes options as a single ``options=`` dict (not
        unpacked kwargs), so we merge into that dict. Caller-supplied
        keys in ``options`` win over configured defaults.
        """
        if not self._options:
            return kwargs
        caller_options = kwargs.get("options") or {}
        return {**kwargs, "options": {**self._options, **caller_options}}

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
                prompt_text, key=self._key, **self._merged_prompt_kwargs(kwargs)
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
        chain_response = self._conversation.chain(
            prompt_text, key=self._key, **self._merged_chain_kwargs(kwargs)
        )
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
        options: Optional[dict] = None,
    ):
        self._model = model
        self._datasette = datasette
        self._purpose = purpose
        self._group = group
        self._key = key
        self._actor = actor
        self._options = options or {}

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
            options=self._options,
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

            # Execute the actual prompt, passing the resolved key.
            # prompt() collects options via **kwargs so configured
            # options merge into the kwargs dict (caller wins).
            merged = {**self._options, **kwargs} if self._options else kwargs
            response = await self._model.prompt(prompt_text, key=self._key, **merged)

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

    def _get_purpose_key_for_model(self, model_id, purpose):
        return ModelLookup(self._get_config(), purpose).setting(model_id, "key")

    def _get_options_for_model(self, model_id, purpose):
        return ModelLookup(self._get_config(), purpose).setting(model_id, "options")

    async def get_key_for_model(
        self,
        model,
        actor: Optional[dict] = None,
        purpose: Optional[str] = None,
        key_name: Optional[str] = None,
        _lookup=None,
    ) -> Optional[str]:
        """
        Resolve API key for a model.

        Explicit names use datasette-secrets, then the same stored LLM key alias.
        Endpoints never inherit provider credentials. Models without an explicit
        name use the standard provider secret and LLM key/environment resolution.

        Args:
            model: The model object (has .needs_key, .key_env_var attributes)
            actor: Optional actor for per-user key resolution
            purpose: Optional purpose for purpose-specific key lookup

        Returns:
            API key string, or None if no key found
        """
        lookup = _lookup or ModelLookup(self._get_config(), purpose)
        key_override = key_name or lookup.setting(model.model_id, "key")
        if hasattr(model, "endpoint_reference"):
            key_override = model.needs_key
        if key_override:
            actor_id = actor.get("id") if actor else None
            key = await get_secret(self._datasette, key_override, actor_id)
            if key:
                return key
            # This is an alias lookup, not explicit_key: a missing alias must
            # never be treated as a literal credential or use a provider key.
            return llm_library.get_key(key_alias=key_override)

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

    async def model_has_key(
        self,
        model,
        actor: Optional[dict] = None,
        purpose: Optional[str] = None,
        key_name: Optional[str] = None,
        _lookup=None,
    ) -> bool:
        """Check if a model has a usable API key configured."""
        lookup = _lookup or ModelLookup(self._get_config(), purpose)
        key_name = key_name or lookup.setting(model.model_id, "key")
        if model.needs_key is None and not key_name:
            return True
        key = await self.get_key_for_model(
            model, actor, purpose=purpose, key_name=key_name, _lookup=lookup
        )
        return bool(key)

    async def _resolve_model_id(
        self,
        model_id: Optional[str],
        purpose: Optional[str],
        actor: Optional[dict],
        _lookup=None,
    ) -> str:
        """
        Resolve the model ID to use.

        Resolution order:
        1. Explicit model_id argument (if provided)
        2. llm_default_model hook
        3. Purpose-specific config: purposes.<purpose>.model
        4. Global default: default_model
        5. LLM's own default model, if available to this actor/purpose
        6. Error if none of the above
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
        lookup = _lookup or ModelLookup(self._get_config(), purpose)
        config = lookup.config

        # Purpose-specific default
        if purpose:
            purposes = config.get("purposes", {})
            if purpose in purposes:
                purpose_model = purposes[purpose].get("model")
                if purpose_model:
                    return ModelReference.parse(purpose_model).model

        # Global default
        default = config.get("default_model")
        if default:
            return ModelReference.parse(default).model

        # Fall back to LLM's own default model - but only if it is
        # actually usable here: installed as an async model, allowed by
        # config/hooks, and has a key (when require_keys is on)
        llm_default = lookup.llm_default_id()
        if llm_default is not None:
            available = await self._available_models(lookup, actor, purpose)
            if llm_default in {m.model_id for m in available}:
                return llm_default

        raise ModelNotFoundError(
            "No model_id provided, no default_model configured, and LLM's "
            "default model ({}) is not available. Either pass a model_id, "
            "configure default_model in datasette-llm settings, or run "
            "'llm models default <model>'.".format(llm_library.get_default_model())
        )

    def _sort_models(self, models, config, purpose):
        return ModelLookup(config, purpose).sort(models)

    async def model(
        self,
        model_id: Optional[str | dict] = None,
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
        inline_ref = (
            ModelReference.parse(model_id) if isinstance(model_id, dict) else None
        )
        if inline_ref and inline_ref.key:
            key_names = getattr(self._datasette, "_llm_inline_key_names", set())
            self._datasette._llm_inline_key_names = key_names | {inline_ref.key}
        lookup = ModelLookup(self._get_config(), purpose, inline_ref)
        resolved_model_id = await self._resolve_model_id(
            inline_ref.model if inline_ref else model_id, purpose, actor, _lookup=lookup
        )
        model = lookup.find(resolved_model_id)
        if model is None:
            raise ModelNotFoundError(f"Unknown model: {resolved_model_id}")

        # Resolve the API key for this model
        key_name = inline_ref.key if inline_ref else None
        key = await self.get_key_for_model(
            model, actor, purpose=purpose, key_name=key_name, _lookup=lookup
        )
        configured_key = key_name or lookup.setting(model.model_id, "key")
        if configured_key and not key:
            raise llm_library.NeedsKeyException(
                f"Key '{configured_key}' for model '{model.model_id}' was not found "
                "in datasette-secrets or LLM's stored keys"
            )

        if model.needs_key and not key and lookup.config.get("require_keys", True):
            # Preserve LLM's actionable provider-specific missing-key message.
            model.get_key()

        available = await self._available_models(lookup, actor, purpose)
        if model.model_id not in {m.model_id for m in available}:
            raise ModelNotFoundError(f"Model '{model.model_id}' is not available")
        if hasattr(model, "endpoint_reference"):
            # Bind the literal credential to this fresh instance, including
            # when Responses delegates to Chat Completions.
            model.key = key

        # Resolve any default options configured for this model
        options = lookup.setting(model.model_id, "options")
        if inline_ref and inline_ref.options is not None:
            options = inline_ref.options

        return WrappedAsyncModel(
            model,
            self._datasette,
            purpose=purpose,
            key=key,
            actor=actor,
            options=options,
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
        lookup = ModelLookup(self._get_config(), purpose)
        return await self._available_models(lookup, actor, purpose)

    async def _available_models(self, lookup, actor, purpose):
        all_models = list(lookup.by_id.values())
        if lookup.config.get("require_keys", True):
            key_available = {}
            with_keys = []
            for model in all_models:
                key_name = lookup.setting(model.model_id, "key")
                cache_key = (key_name, model.needs_key, model.key_env_var)
                if cache_key not in key_available:
                    key_available[cache_key] = await self.model_has_key(
                        model, actor, purpose=purpose, key_name=key_name, _lookup=lookup
                    )
                if key_available[cache_key]:
                    with_keys.append(model)
            all_models = with_keys

        # Each hook sees the same initial list; intersect all restrictions.
        token = current_lookup.set((self._datasette, lookup))
        try:
            for result in pm.hook.llm_filter_models(
                datasette=self._datasette,
                models=all_models,
                actor=actor,
                purpose=purpose,
            ):
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None:
                    allowed_ids = {model.model_id for model in result}
                    all_models = [
                        model for model in all_models if model.model_id in allowed_ids
                    ]
        finally:
            current_lookup.reset(token)
        return lookup.sort(all_models)

    @asynccontextmanager
    async def group(
        self,
        model_id: Optional[str | dict] = None,
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
        wrapped = await self.model(model_id, purpose=purpose, actor=actor)
        group_obj = Group(model_id=wrapped.model_id, purpose=purpose, actor=actor)
        wrapped._group = group_obj

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
    "ModelConfigurationError",
    "get_purposes",
]
