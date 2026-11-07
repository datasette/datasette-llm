# Changelog

## Unreleased

- Model-reference dictionaries can define OpenAI-compatible Chat Completions or Responses endpoints with tool support. Use `endpoints` to register additional models without enabling an allowlist; dictionaries also work in defaults, model lists, `model()` and `group()`.
- An explicitly named `key` is resolved through datasette-secrets and then the same stored LLM key alias. Endpoint credentials remain isolated, including when LLM routes a Responses request to Chat Completions.
- Model lookup takes one async model/alias snapshot per call and precomputes configured key and option overrides. Registry work no longer grows with installed models multiplied by configuration entries. Snapshots are local to each call so plugin and alias changes remain visible.
- Invalid model-reference fields no longer prevent other plugins from looking up secrets. Endpoint definitions with conflicting IDs are omitted from listings and raise `ModelConfigurationError` only when selected, leaving unrelated installed models usable.
- Missing provider keys raise LLM's actionable `NeedsKeyException`, including instructions for setting a stored key or environment variable.
- Use `key` for credentials. Extra `headers` can appear unredacted in Datasette's public `/-/config.json` output.

### Compatibility changes

- An explicit `key` on an installed model no longer falls back to the provider's default credential. If that name is missing from datasette-secrets and LLM's stored keys, selecting the model raises `NeedsKeyException`, and it is omitted from listings when `require_keys` is true. Remove `key` to restore normal provider key resolution.
- Allowlists, blocklists and `llm_filter_models` hooks now apply to `model()` and `group()`, including explicit IDs and configured defaults. Selecting an excluded model raises `ModelNotFoundError`.
- A purpose's model allowlist also includes the global `default_model`. Global and purpose lists remain combined, and blocklists and filter hooks can still exclude those defaults.
- Model-reference dictionaries now reject unknown fields and explicit `key: null` or `options: null` values with `ModelConfigurationError`. Omit those optional fields instead. Invalid references are ignored while registering secrets so they cannot disable other plugins' secrets.
- Unknown model IDs raise `datasette_llm.ModelNotFoundError` instead of `llm.UnknownModelError`.
- This release requires LLM 0.37.1 or newer.
