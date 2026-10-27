"""Feature C: the LangGraph chat workflow, run through /chat and directly."""

import asyncio

from neo4j_agent_memory import TransportError

from app.graph.workflow import build_chat_graph
from tests.conftest import make_entity

ATLAS = ("Project Atlas is a satellite imaging platform built by the geo team. " * 6).encode()


def new_conversation(client) -> str:
    return client.post("/conversations", json={"title": "t"}).json()["conversation_id"]


def chat(client, conversation_id, message="What is Project Atlas?"):
    return client.post("/chat", json={"conversation_id": conversation_id, "message": message})


def test_nodes_run_in_the_expected_order(settings, vector, nams, llm, sdk):
    sdk.short_term.conversations["c1"] = []
    graph = build_chat_graph(vector, nams, llm, settings)

    async def run():
        order = []
        async for update in graph.astream({"conversation_id": "c1", "user_message": "hi"}, stream_mode="updates"):
            order.extend(update.keys())
        return order

    assert asyncio.run(run()) == [
        "retrieve_document_context",
        "retrieve_memory_context",
        "generate_answer",
        "save_conversation",
    ]


def test_chat_uses_documents_memory_and_saves(client, sdk, llm):
    client.post("/documents/upload", files={"file": ("atlas.md", ATLAS, "text/markdown")})
    sdk.long_term.entities = [make_entity("Ahmed", "PERSON", "Works on Project Atlas")]
    conv = new_conversation(client)

    body = chat(client, conv).json()

    assert body["answer"] == "Fake answer." and body["conversation_id"] == conv
    assert body["document_sources"] == ["atlas.md"]
    assert body["memory_retrieval_succeeded"] is True and body["memory_retrieval_status"] == "ok"
    assert body["memory_entities_used"][0]["name"] == "Ahmed"
    assert body["messages_saved"] is True
    # saving is not extraction: NAMS is still working in the background
    assert body["extraction_status"] == "pending"

    call = llm.calls[0]
    assert call["chunks"] and call["entities"][0]["name"] == "Ahmed"
    saved = [(m.role.value, m.content) for m in sdk.short_term.conversations[conv]]
    assert saved == [("user", "What is Project Atlas?"), ("assistant", "Fake answer.")]


def test_second_turn_receives_the_saved_history(client, llm):
    conv = new_conversation(client)
    chat(client, conv, "My name is Sara")
    chat(client, conv, "What is my name?")
    assert llm.calls[1]["history"] == [
        {"role": "user", "content": "My name is Sara"},
        {"role": "assistant", "content": "Fake answer."},
    ]


def test_chat_without_any_context_still_answers(client, llm):
    body = chat(client, new_conversation(client)).json()
    assert body["document_sources"] == [] and body["memory_retrieval_status"] == "empty"
    assert llm.calls[0]["chunks"] == [] and llm.calls[0]["entities"] == []


def test_memory_failure_degrades_gracefully(client, sdk):
    conv = new_conversation(client)
    sdk.long_term.error = TransportError("down")
    body = chat(client, conv).json()
    assert body["answer"] == "Fake answer."
    assert body["memory_retrieval_succeeded"] is False and body["memory_retrieval_status"] == "unavailable"
    assert body["memory_entities_used"] == []
    assert body["messages_saved"] is True
    assert any("Long-term memory unavailable" in w for w in body["warnings"])


def test_save_failure_is_reported_not_hidden(client, sdk):
    conv = new_conversation(client)

    original = sdk.short_term.bulk_add_messages

    async def failing(*args, **kwargs):
        raise TransportError("down")

    sdk.short_term.bulk_add_messages = failing
    body = chat(client, conv).json()
    sdk.short_term.bulk_add_messages = original
    assert body["answer"] == "Fake answer."
    assert body["messages_saved"] is False and body["extraction_status"] == "not_checked"
    assert any("NOT saved" in w for w in body["warnings"])


def test_unknown_conversation_is_404(client, llm):
    response = chat(client, "does-not-exist")
    assert response.status_code == 404 and "POST /conversations" in response.json()["detail"]
    assert llm.calls == []  # the LLM is never called


def test_chat_does_not_wait_for_extraction_by_default(client, sdk):
    chat(client, new_conversation(client))
    assert sdk.long_term.wait_calls == []


def test_optional_bounded_wait_for_extraction(settings, vector, nams, llm, sdk):
    sdk.short_term.conversations["c1"] = []
    sdk.short_term.extraction_summary = {"completed": 2}
    waiting = settings.model_copy(update={"extraction_wait_seconds": 3})
    graph = build_chat_graph(vector, nams, llm, waiting)
    state = asyncio.run(graph.ainvoke({"conversation_id": "c1", "user_message": "hi"}))
    assert sdk.long_term.wait_calls == [3]  # bounded by the configured timeout
    assert state["extraction_status"] == "completed"


def test_chat_validates_request(client):
    assert client.post("/chat", json={"conversation_id": "c", "message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "hi"}).status_code == 422
