"""Optional smoke test against the REAL NAMS service. Skipped unless explicitly enabled.

    $env:RUN_INTEGRATION = "1"; python -m pytest tests/test_integration.py

It needs MEMORY_API_KEY (in .env or the environment). It creates one conversation in your NAMS
workspace and does not call the LLM or the embedding API, so it costs no OpenAI credits.
"""

import asyncio
import os

import pytest

from app.config import Settings
from app.services.nams_service import NamsService

settings = Settings()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1" or settings.memory_api_key is None,
        reason="set RUN_INTEGRATION=1 and MEMORY_API_KEY to run against the real NAMS service",
    ),
]


def test_real_nams_roundtrip():
    async def run():
        service = NamsService(settings)
        try:
            conversation_id = await service.create_conversation("memoryrag-integration-test")
            await service.save_exchange(conversation_id, "My name is Test User.", "Nice to meet you.")
            history = await service.get_history(conversation_id, 10)
            status = await service.extraction_status(conversation_id)
            entities = await service.search_entities("Test User", 3)
            return history, status, entities
        finally:
            await service.aclose()

    history, status, entities = asyncio.run(run())
    assert [t["role"] for t in history] == ["user", "assistant"]
    assert status["state"] in {"pending", "completed", "failed", "unknown"}
    assert isinstance(entities, list)
