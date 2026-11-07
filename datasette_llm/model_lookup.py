from contextvars import ContextVar

import llm

from .model_refs import ModelConfigurationError, ModelReference


current_lookup = ContextVar("datasette_llm_model_lookup", default=None)


def iter_model_refs(config):
    if config.get("default_model"):
        yield config["default_model"]
    yield from config.get("models", [])
    yield from config.get("endpoints", [])
    for purpose_config in config.get("purposes", {}).values():
        if purpose_config.get("model"):
            yield purpose_config["model"]
        yield from purpose_config.get("models", [])
        yield from purpose_config.get("endpoints", [])


class ModelLookup:
    """One operation's registry snapshot and normalized configuration."""

    def __init__(self, config, purpose=None, inline_ref=None):
        self.config = config
        self.purpose = purpose
        self.aliases = llm.get_async_model_aliases()
        self.purpose_config = (
            config.get("purposes", {}).get(purpose, {}) if purpose else {}
        )
        pc = self.purpose_config
        raw_refs = list(pc.get("models", []))
        if pc.get("model"):
            raw_refs.append(pc["model"])
        raw_refs.extend(pc.get("endpoints", []))
        if config.get("default_model"):
            raw_refs.append(config["default_model"])
        raw_refs.extend(config.get("models", []))
        raw_refs.extend(config.get("endpoints", []))
        self.refs = [ModelReference.parse(ref) for ref in raw_refs]
        if inline_ref:
            self.refs.insert(0, inline_ref)
        self.endpoint_ids = {ref.model for ref in self.refs if ref.api_base}
        self.settings = {}
        for ref in self.refs:
            settings = self.settings.setdefault(self.canonical_id(ref.model), {})
            for name in ("key", "options"):
                value = getattr(ref, name)
                if value is not None:
                    settings.setdefault(name, value)

        self.errors = {}
        definitions = {}
        all_refs = [ModelReference.parse(ref) for ref in iter_model_refs(config)]
        if inline_ref:
            all_refs.append(inline_ref)
        for ref in all_refs:
            if not ref.api_base:
                continue
            previous = definitions.get(ref.model)
            if previous and previous.endpoint_identity() != ref.endpoint_identity():
                self.errors[ref.model] = (
                    f"Conflicting endpoint definitions for model '{ref.model}'"
                )
            definitions[ref.model] = ref

        self.by_id = {model.model_id: model for model in self.aliases.values()}
        added = set()
        for ref in self.refs:
            if not ref.api_base or ref.model in added:
                continue
            added.add(ref.model)
            if ref.model in self.aliases:
                self.errors[ref.model] = (
                    f"Endpoint model '{ref.model}' conflicts with an installed model or alias; "
                    "choose another model ID and set model_name to the remote model ID"
                )
            if ref.model not in self.errors:
                self.by_id[ref.model] = ref.create_model(self.setting(ref.model, "key"))
            else:
                self.by_id.pop(ref.model, None)

    def canonical_id(self, model_id):
        # An endpoint reference must never resolve to an installed alias: that
        # could apply its credentials or options to a different provider.
        if model_id in self.endpoint_ids:
            return model_id
        model = self.aliases.get(model_id)
        return model.model_id if model else model_id

    def ref_id(self, ref):
        return self.canonical_id(ModelReference.parse(ref).model)

    def setting(self, model_id, name):
        return self.settings.get(model_id, {}).get(name)

    def find(self, model_id):
        if model_id in self.endpoint_ids and model_id in self.errors:
            raise ModelConfigurationError(self.errors[model_id])
        return self.by_id.get(self.canonical_id(model_id))

    def llm_default_id(self):
        model = self.aliases.get(llm.get_default_model())
        return model.model_id if model else None

    def filter_config(self, models):
        config, pc = self.config, self.purpose_config
        allowed = config.get("models")
        purpose_allowed = pc.get("models")
        if allowed or purpose_allowed:
            allowed_ids = {self.ref_id(ref) for ref in allowed or []}
            if self.purpose:
                allowed_ids.update(self.ref_id(ref) for ref in purpose_allowed or [])
                if pc.get("model"):
                    allowed_ids.add(self.ref_id(pc["model"]))
            if config.get("default_model"):
                allowed_ids.add(self.ref_id(config["default_model"]))
            models = [model for model in models if model.model_id in allowed_ids]
        blocked = {self.canonical_id(name) for name in config.get("blocked_models", [])}
        blocked.update(self.canonical_id(name) for name in pc.get("blocked_models", []))
        return [model for model in models if model.model_id not in blocked]

    def sort(self, models):
        default = self.purpose_config.get("model") or self.config.get("default_model")
        default_id = self.ref_id(default) if default else self.llm_default_id()
        ids = [default_id]
        ids.extend(self.ref_id(ref) for ref in self.purpose_config.get("models", []))
        ids.extend(self.ref_id(ref) for ref in self.config.get("models", []))
        ids.extend(model.model_id for model in models)
        by_id = {model.model_id: model for model in models}
        result = []
        for model_id in ids:
            model = by_id.pop(model_id, None)
            if model is not None:
                result.append(model)
        return result
