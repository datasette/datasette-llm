from copy import copy
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import llm
from llm.default_plugins.openai_models import AsyncChat, AsyncResponses


class ModelConfigurationError(ValueError):
    """An invalid or conflicting model reference."""


@dataclass(frozen=True)
class ModelReference:
    model: str
    key: str | None = None
    options: dict | None = None
    api_base: str | None = None
    model_name: str | None = None
    responses: bool = False
    supports_tools: bool = True
    can_stream: bool = True
    supports_schema: bool = False
    vision: bool = False
    audio: bool = False
    reasoning: bool = False
    headers: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, value):
        if isinstance(value, str):
            value = {"model": value}
        if not isinstance(value, dict):
            raise ModelConfigurationError(
                "A model reference must be a string or dictionary"
            )
        unknown = value.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ModelConfigurationError(
                "Unknown model reference fields: {}".format(
                    ", ".join(sorted(map(str, unknown)))
                )
            )
        for name in ("model", "key", "api_base", "model_name"):
            if name in value and (
                not isinstance(value[name], str) or not value[name].strip()
            ):
                raise ModelConfigurationError(f"{name} must be a non-empty string")
        if "model" not in value:
            raise ModelConfigurationError("A model reference requires a model ID")
        if "options" in value and not isinstance(value["options"], dict):
            raise ModelConfigurationError("options must be a dictionary")
        for name in (
            "responses",
            "supports_tools",
            "can_stream",
            "supports_schema",
            "vision",
            "audio",
            "reasoning",
        ):
            if name in value and not isinstance(value[name], bool):
                raise ModelConfigurationError(f"{name} must be a boolean")
        headers = value.get("headers", {})
        if not isinstance(headers, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in headers.items()
        ):
            raise ModelConfigurationError("headers must be a dictionary of strings")
        endpoint_fields = value.keys() - {"model", "key", "options"}
        if endpoint_fields and "api_base" not in value:
            raise ModelConfigurationError("Endpoint fields require api_base")
        if "api_base" in value:
            try:
                url = urlsplit(value["api_base"])
                valid = (
                    url.scheme in ("http", "https")
                    and url.hostname
                    and url.username is None
                    and url.password is None
                    and not url.query
                    and not url.fragment
                )
                _ = url.port  # Validate the port, if present
            except ValueError:
                valid = False
            if not valid:
                raise ModelConfigurationError(
                    "api_base must be an HTTP(S) base URL without credentials, query or fragment"
                )
            value = {**value, "api_base": value["api_base"].rstrip("/")}
        return cls(**value)

    def endpoint_kwargs(self):
        kwargs = {
            "model_id": self.model,
            "model_name": self.model_name or self.model,
            "api_base": self.api_base,
            "supports_tools": self.supports_tools,
            "can_stream": self.can_stream,
            "supports_schema": self.supports_schema,
            "vision": self.vision,
            "audio": self.audio,
            "reasoning": self.reasoning,
            "headers": self.headers,
        }
        if self.responses:
            kwargs["reasoning_summary"] = False
        return kwargs

    def endpoint_identity(self):
        return self.responses, self.endpoint_kwargs()

    def create_model(self, key_name=None):
        klass = EndpointAsyncResponses if self.responses else EndpointAsyncChat
        model = klass(**self.endpoint_kwargs())
        model.endpoint_reference = self
        model.needs_key = key_name
        model.key_env_var = None
        return model


class _EndpointCredentials:
    async def execute(self, prompt, stream, response, conversation=None, key=None):
        # Bind a literal credential on an execution-local model. LLM's own
        # Responses->Chat routing propagates model.key to its delegated model.
        # A dummy credential prevents that path from consulting provider keys.
        bound = copy(self)
        bound.key = self.get_key(key) if self.needs_key else "DUMMY_KEY"
        async for event in super(_EndpointCredentials, bound).execute(
            prompt, stream and self.can_stream, response, conversation, bound.key
        ):
            yield event

    def get_key(self, explicit_key=None):
        # Resolved secret values are literal credentials, never LLM key aliases.
        if explicit_key is not None:
            return explicit_key
        if self.key is not None:
            return self.key
        if self.needs_key is not None:
            raise llm.NeedsKeyException(f"Missing key: {self.needs_key}")
        return None


class EndpointAsyncChat(_EndpointCredentials, AsyncChat):
    pass


class EndpointAsyncResponses(_EndpointCredentials, AsyncResponses):
    pass
