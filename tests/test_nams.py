"""Features B, D, E and the NAMS service: conversations, memory search, health, failures, async extraction."""

import pytest
from neo4j_agent_memory import AuthenticationError, RateLimitError, TransportError

from app.config import Settings
from app.errors import ConfigError, UpstreamError
from app.services.nams_service import NamsService, classify_extraction, to_upstream_error
from tests.conftest import make_entity


# --- health -----------------------------------------------------------------------------------


def test_health_ok_and_hides_secrets(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["nams"] == {"configured": True}
    assert "nams_test_key" not in response.text and "sk-test-llm-key" not in response.text


def test_health_reports_degraded_without_credentials(tmp_path, vector, llm):
    from fastapi.testclient import TestClient

    from app.main import create_app

    bare = Settings(_env_file=None, memory_api_key=None, llm_api_key=None, chroma_dir=str(tmp_path / "c"))
    llm.is_configured = False
    with TestClient(create_app(bare, vector=vector, nams=NamsService(bare), llm=llm)) as c:
        body = c.get("/health").json()
    assert body["status"] == "degraded"
    assert body["nams"]["configured"] is False and body["llm"]["configured"] is False


def test_deep_health_checks_nams(client, sdk):
    assert client.get("/health", params={"deep": True}).json()["nams"]["reachable"] is True
    sdk.short_term.error = TransportError("down")
    body = client.get("/health", params={"deep": True}).json()
    assert body["status"] == "degraded" and body["nams"]["reachable"] is False


# --- conversations ----------------------------------------------------------------------------


def test_create_conversation_returns_the_nams_id(client, sdk):
    response = client.post("/conversations", json={"title": "AI Research"})
    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "AI Research"
    assert body["conversation_id"] in sdk.short_term.conversations  # the id NAMS generated
    assert sdk.short_term.created_with[0]["metadata"] == {"title": "AI Research"}


def test_create_conversation_validates_title(client):
    assert client.post("/conversations", json={"title": ""}).status_code == 422
    assert client.post("/conversations", json={"title": "   "}).status_code == 422


def test_create_conversation_bad_credentials_gives_safe_error(client, sdk):
    sdk.short_term.error = AuthenticationError("401 for key nams_test_key")
    response = client.post("/conversations", json={"title": "x"})
    assert response.status_code == 502
    assert "MEMORY_API_KEY" in response.json()["detail"]
    assert "nams_test_key" not in response.text  # raw SDK message is never echoed


def test_missing_memory_key_gives_a_helpful_error(tmp_path, vector, llm, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.delenv("MEMORY_API_KEY", raising=False)
    bare = Settings(_env_file=None, memory_api_key=None, llm_api_key="k", chroma_dir=str(tmp_path / "c"))
    with TestClient(create_app(bare, vector=vector, nams=NamsService(bare), llm=llm)) as c:
        response = c.post("/conversations", json={"title": "x"})
    assert response.status_code == 503 and "MEMORY_API_KEY" in response.json()["detail"]


@pytest.mark.parametrize(
    "error, status",
    [(AuthenticationError("x"), 502), (RateLimitError("x"), 429), (TransportError("x"), 503), (RuntimeError("x"), 502)],
)
def test_error_translation_statuses(error, status):
    assert to_upstream_error(error, "do something").status_code == status


# --- asynchronous extraction -----------------------------------------------------------------


def test_classify_extraction():
    assert classify_extraction({"pending": 1, "completed": 3}) == "pending"
    assert classify_extraction({"completed": 2}) == "completed"
    assert classify_extraction({"completed": 1, "failed": 1}) == "failed"
    # NAMS reporting nothing must NOT be treated as success
    assert classify_extraction({}) == "unknown"


def test_extraction_status_endpoint_follows_nams(client, sdk):
    conv = client.post("/conversations", json={"title": "t"}).json()["conversation_id"]
    url = f"/conversations/{conv}/extraction-status"
    assert client.get(url).json()["state"] == "pending"
    sdk.short_term.extraction_summary = {"completed": 2}
    body = client.get(url).json()
    assert body["state"] == "completed" and body["pending_messages"] == 0


# --- memory search ----------------------------------------------------------------------------


def test_memory_search_returns_entities_with_their_own_fields(client, sdk):
    sdk.long_term.entities = [
        make_entity("Ahmed", "PERSON", "Works on Project Atlas"),
        make_entity("Atlas", "OBJECT", ""),  # no description: the key is simply absent
    ]
    body = client.get("/memory/search", params={"q": "Ahmed", "limit": 5}).json()
    assert body["count"] == 2 and body["query"] == "Ahmed"
    first, second = body["results"]
    assert first["name"] == "Ahmed" and first["description"] == "Works on Project Atlas"
    assert "description" not in second
    assert "embedding" not in first
    assert sdk.long_term.queries == ["Ahmed"]


def test_memory_search_without_matches_explains_extraction(client):
    body = client.get("/memory/search", params={"q": "nobody"}).json()
    assert body["count"] == 0 and body["results"] == []
    assert "extraction" in body["note"].lower()


@pytest.mark.parametrize("params", [{"q": ""}, {"q": "x" * 201}, {"q": "a", "limit": 0}, {"q": "a", "limit": 26}, {}])
def test_memory_search_validates_input(client, params):
    assert client.get("/memory/search", params=params).status_code == 422


def test_memory_search_nams_unreachable(client, sdk):
    sdk.long_term.error = TransportError("down")
    response = client.get("/memory/search", params={"q": "Ahmed"})
    assert response.status_code == 503


def test_service_raises_config_error_without_key():
    service = NamsService(Settings(_env_file=None, memory_api_key=None))
    import asyncio

    with pytest.raises(ConfigError):
        asyncio.run(service.search_entities("x", 1))
    assert isinstance(to_upstream_error(RuntimeError(), "x"), UpstreamError)
