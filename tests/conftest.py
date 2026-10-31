import pytest


@pytest.fixture(autouse=True)
def isolated_llm_user_dir(tmp_path, monkeypatch):
    # Stop the developer's own LLM config (default_model.txt, keys.json,
    # aliases) leaking into tests
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path / "llm"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
