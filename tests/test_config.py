from src.config import load_streamlit_secrets


def test_streamlit_secrets_populate_missing_environment(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    load_streamlit_secrets({"DATABASE_URL": "postgresql://example"})
    assert __import__("os").environ["DATABASE_URL"] == "postgresql://example"


def test_environment_overrides_streamlit_secret(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite://local")
    load_streamlit_secrets({"DATABASE_URL": "postgresql://example"})
    assert __import__("os").environ["DATABASE_URL"] == "sqlite://local"
