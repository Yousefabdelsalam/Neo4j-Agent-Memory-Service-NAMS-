"""The web UI and the two small endpoints it relies on."""


def test_root_serves_the_web_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"] and "MemoryRAG" in response.text


def test_static_assets_are_served(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_ui_is_not_listed_in_the_api_schema(client):
    assert "/" not in client.get("/openapi.json").json()["paths"]


def test_list_documents_groups_chunks_per_document(client):
    assert client.get("/documents").json() == []
    text = ("Project Atlas is a satellite imaging platform. " * 20).encode()
    client.post("/documents/upload", files={"file": ("b.md", text, "text/markdown")}, data={"document_id": "b"})
    client.post("/documents/upload", files={"file": ("a.txt", b"short note", "text/plain")}, data={"document_id": "a"})
    docs = client.get("/documents").json()
    assert [d["filename"] for d in docs] == ["a.txt", "b.md"]  # sorted by name
    assert docs[0] == {"document_id": "a", "filename": "a.txt", "chunks": 1}
    assert docs[1]["chunks"] > 1


def test_conversation_messages_returns_saved_turns(client):
    conv = client.post("/conversations", json={"title": "t"}).json()["conversation_id"]
    assert client.get(f"/conversations/{conv}/messages").json()["messages"] == []
    client.post("/chat", json={"conversation_id": conv, "message": "Hello"})
    messages = client.get(f"/conversations/{conv}/messages").json()["messages"]
    assert messages == [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Fake answer."}]


def test_conversation_messages_unknown_id_is_404(client):
    assert client.get("/conversations/nope/messages").status_code == 404
