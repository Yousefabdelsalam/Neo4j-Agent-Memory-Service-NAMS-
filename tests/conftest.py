"""Shared test fixtures.

Everything fake lives in this folder: a stand-in for the NAMS SDK client, a deterministic embedding
function and a stand-in LLM. The application code itself contains no mocks.
"""

import hashlib
import math
import os
import re
import tempfile
import uuid

# app.main builds a module-level `app` on import; keep its Chroma data out of the real data dir.
os.environ["CHROMA_DIR"] = tempfile.mkdtemp(prefix="memoryrag-import-")

import pytest
from fastapi.testclient import TestClient
from neo4j_agent_memory import NotFoundError
from neo4j_agent_memory.memory.long_term import Entity
from neo4j_agent_memory.memory.short_term import Conversation, Message
from neo4j_agent_memory.nams.short_term import ExtractionStatus

from app.config import Settings
from app.main import create_app
from app.services.nams_service import NamsService
from app.services.vector_service import VectorService


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words embedding: texts sharing words get similar vectors."""
    vectors = []
    for text in texts:
        vec = [0.0] * 128
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 128] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vectors.append([x / norm for x in vec])
    return vectors


class FakeShortTerm:
    def __init__(self):
        self.conversations: dict[str, list[Message]] = {}
        self.created_with: list[dict] = []
        self.extraction_summary = {"pending": 2}
        self.error: Exception | None = None  # raised by every call when set

    def _maybe_fail(self):
        if self.error:
            raise self.error

    async def create_conversation(self, session_id=None, **kwargs):
        self._maybe_fail()
        conv_id = uuid.uuid4()
        self.conversations[str(conv_id)] = []
        self.created_with.append({"label": session_id, **kwargs})
        return Conversation(id=conv_id, session_id=session_id or "x")

    async def get_conversation(self, session_id=None, **kwargs):
        self._maybe_fail()
        if session_id not in self.conversations:
            raise NotFoundError("no such conversation")
        return Conversation(id=uuid.uuid4(), session_id=session_id, messages=list(self.conversations[session_id]))

    async def bulk_add_messages(self, session_id=None, messages=None, **kwargs):
        self._maybe_fail()
        if session_id not in self.conversations:
            raise NotFoundError("no such conversation")
        self.conversations[session_id].extend(Message(role=m["role"], content=m["content"]) for m in messages)
        return []

    async def get_extraction_status(self, session_id=None, **kwargs):
        self._maybe_fail()
        return ExtractionStatus(summary=self.extraction_summary)

    async def list_conversations(self, **kwargs):
        self._maybe_fail()
        return []


class FakeLongTerm:
    def __init__(self):
        self.entities: list[Entity] = []
        self.error: Exception | None = None
        self.wait_calls: list[float] = []
        self.queries: list[str] = []

    async def search_entities(self, query, **kwargs):
        if self.error:
            raise self.error
        self.queries.append(query)
        return self.entities[: kwargs.get("limit", 5)]

    async def wait_for_extraction(self, *, session_id=None, timeout=30.0, **kwargs):
        self.wait_calls.append(timeout)
        return True


class FakeSdkClient:
    """Mimics the parts of neo4j_agent_memory.MemoryClient (NAMS backend) that the app uses."""

    def __init__(self):
        self.short_term = FakeShortTerm()
        self.long_term = FakeLongTerm()
        self.closed = False

    async def close(self):
        self.closed = True


class FakeLLM:
    is_configured = True

    def __init__(self):
        self.calls: list[dict] = []
        self.reply = "Fake answer."

    async def answer(self, user_message, history, chunks, entities):
        self.calls.append({"user_message": user_message, "history": history, "chunks": chunks, "entities": entities})
        return self.reply


def make_entity(name="Ahmed", type_="PERSON", description="Works on Project Atlas") -> Entity:
    return Entity(name=name, type=type_, description=description)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        memory_api_key="nams_test_key",
        llm_api_key="sk-test-llm-key",
        chroma_dir=str(tmp_path / "chroma"),
        chunk_size=200,
        chunk_overlap=20,
        max_upload_mb=0.01,  # ~10 KB, so the "too large" test stays small
    )


@pytest.fixture
def sdk():
    return FakeSdkClient()


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def vector(settings):
    return VectorService(settings, embed=fake_embed)


@pytest.fixture
def nams(settings, sdk):
    return NamsService(settings, client=sdk)


@pytest.fixture
def client(settings, vector, nams, llm):
    with TestClient(create_app(settings, vector=vector, nams=nams, llm=llm)) as test_client:
        yield test_client
