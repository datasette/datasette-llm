# datasette-llm

[![PyPI](https://img.shields.io/pypi/v/datasette-llm.svg)](https://pypi.org/project/datasette-llm/)
[![Changelog](https://img.shields.io/github/v/release/datasette/datasette-llm?include_prereleases&label=changelog)](https://github.com/datasette/datasette-llm/releases)
[![Tests](https://github.com/datasette/datasette-llm/actions/workflows/test.yml/badge.svg)](https://github.com/datasette/datasette-llm/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/datasette/datasette-llm/blob/main/LICENSE)

LLM integration for Datasette plugins.

This plugin provides a standard interface for Datasette plugins to use LLM models via the [llm](https://llm.datasette.io/) library, with:

- **Model management**: Control which models are available, with filtering and defaults
- **API key management**: Integration with [datasette-secrets](https://github.com/datasette/datasette-secrets) for secure key storage
- **Hooks for extensibility**: Track usage, enforce policies, implement accounting

## Installation

Install this plugin in the same environment as Datasette:

```bash
datasette install datasette-llm
```

You'll also need at least one LLM model plugin installed:

```bash
# For OpenAI models
datasette install llm

# For Anthropic models
datasette install llm-anthropic

# For testing without API calls
datasette install llm-echo
```

## Configuration

Configure the plugin in your `datasette.yaml`:

```yaml
plugins:
  datasette-llm:
    # Default model when none specified
    default_model: gpt-4o-mini

    # Purpose-specific model defaults
    purposes:
      enrichments:
        model: gpt-4o-mini      # Cheap for bulk operations
      sql-assistant:
        model: gpt-4o           # Smarter for complex queries
      chat:
        model: claude-3-5-sonnet

    # Model availability (optional)
    models:                      # Allowlist - only these models available
      - gpt-4o-mini
      - gpt-4o
      - claude-3-5-sonnet

    # Or use a blocklist instead
    blocked_models:
      - o1-preview              # Too expensive

    # Only show models with API keys configured (default: true)
    require_keys: true
```

## API Key Management

datasette-llm integrates with [datasette-secrets](https://github.com/datasette/datasette-secrets) for API key management. Keys are automatically registered for all installed model providers.

### Setting up keys

1. **Via environment variables** (recommended for deployment):
   ```bash
   export DATASETTE_SECRETS_OPENAI_API_KEY=sk-...
   export DATASETTE_SECRETS_ANTHROPIC_API_KEY=sk-ant-...
   ```

2. **Via the web interface**: Navigate to `/-/secrets` (requires `manage-secrets` permission)

3. **Via llm CLI** (fallback): Keys set with `llm keys set openai` are also used

### Key resolution order

1. datasette-secrets (env var `DATASETTE_SECRETS_<PROVIDER>_API_KEY` or encrypted database)
2. llm's keys.json (`~/.config/io.datasette.llm/keys.json`)
3. llm's environment variables (e.g., `OPENAI_API_KEY`)

## Usage

### Basic usage

```python
from datasette_llm import LLM

async def my_plugin_view(datasette, request):
    llm = LLM(datasette)

    # Get a model (uses default if configured)
    model = await llm.model()

    # Or specify a model explicitly
    model = await llm.model("gpt-4o-mini")

    # Execute a prompt
    response = await model.prompt("What is the capital of France?")
    text = await response.text()
```

### The `purpose` parameter

Specify a `purpose` to:
- Select the right default model for the task
- Enable purpose-based auditing and permissions
- Allow purpose-specific budget limits (via datasette-llm-accountant)

```python
# Uses the model configured for "sql-assistant" purpose
model = await llm.model(purpose="sql-assistant")

# Or with explicit model (purpose still tracked)
model = await llm.model("gpt-4o", purpose="sql-assistant")
```

### Streaming responses

```python
model = await llm.model("gpt-4o-mini")
response = await model.prompt("Tell me a story")

# Non-streaming - wait for complete response
text = await response.text()

# Streaming - process chunks as they arrive
async for chunk in response:
    print(chunk, end="", flush=True)
```

### Grouping prompts

Use `group()` for batch operations where multiple prompts are logically related:

```python
async def enrich_rows(datasette, rows):
    llm = LLM(datasette)

    # Model determined by purpose configuration
    async with llm.group(purpose="enrichments") as model:
        results = []
        for row in rows:
            response = await model.prompt(f"Summarize: {row['content']}")
            text = await response.text()
            results.append(text)

    # All responses guaranteed complete here
    return results
```

Benefits of `group()`:
- **Transactional semantics**: All responses forced to complete on exit
- **Shared context**: Hooks can treat grouped prompts together (e.g., shared budget reservation)
- **Cleanup**: The `llm_group_exit` hook is called for settlement/logging

### Listing available models

```python
llm = LLM(datasette)

# Get all available models (filtered by config and key availability)
models = await llm.models()
for model in models:
    print(model.model_id)

# Filter by actor (for per-user permissions)
models = await llm.models(actor=request.actor)

# Filter by purpose
models = await llm.models(purpose="enrichments")
```

## Plugin Hooks

datasette-llm provides hooks for other plugins to extend LLM operations.

### `llm_prompt_context`

Wrap prompt execution with custom logic:

```python
from datasette import hookimpl
from contextlib import asynccontextmanager

@hookimpl
def llm_prompt_context(datasette, model_id, prompt, purpose):
    @asynccontextmanager
    async def wrapper(result):
        # Before the prompt executes
        print(f"Starting prompt to {model_id}")

        yield

        # After prompt() returns (response may still be streaming)
        async def on_complete(response):
            usage = await response.usage()
            print(f"Used {usage.input} input, {usage.output} output tokens")

        if result.response:
            await result.response.on_done(on_complete)

    return wrapper
```

### `llm_group_exit`

Called when a `group()` context manager exits:

```python
@hookimpl
def llm_group_exit(datasette, group):
    # Can return a coroutine for async cleanup
    async def cleanup():
        print(f"Group for {group.purpose} completed")
        print(f"Processed {len(group._responses)} prompts")
    return cleanup()
```

### `llm_filter_models`

Filter the list of available models:

```python
@hookimpl
async def llm_filter_models(datasette, models, actor, purpose):
    if not actor:
        # Anonymous users get limited models
        return [m for m in models if m.model_id == "gpt-4o-mini"]

    # Check database for user's allowed models
    db = datasette.get_database()
    result = await db.execute(
        "SELECT model_id FROM user_models WHERE user_id = ?",
        [actor["id"]]
    )
    allowed = {row["model_id"] for row in result.rows}
    return [m for m in models if m.model_id in allowed]
```

### `llm_default_model`

Provide dynamic default model selection:

```python
@hookimpl
async def llm_default_model(datasette, purpose, actor):
    if actor:
        # Check user's preferred model
        db = datasette.get_database()
        result = await db.execute(
            "SELECT preferred_model FROM user_prefs WHERE user_id = ?",
            [actor["id"]]
        )
        row = result.first()
        if row:
            return row["preferred_model"]
    return None  # Use config defaults
```

## Related Plugins

- **[datasette-secrets](https://github.com/datasette/datasette-secrets)**: Secure API key storage (required dependency)
- **[datasette-llm-accountant](https://github.com/datasette/datasette-llm-accountant)**: Budget management and cost tracking

## Development

To set up this plugin locally:

```bash
cd datasette-llm
uv sync

# Confirm the plugin is visible
uv run datasette plugins
```

To run the tests:

```bash
uv run pytest
```

The test suite uses the `llm-echo` model which echoes back prompts without making API calls.
