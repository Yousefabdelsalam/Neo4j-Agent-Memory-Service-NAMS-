"""Everything that talks to NAMS (Neo4j Agent Memory Service) lives in this file.

Read this file first if you want to learn how NAMS works. The SDK calls used here were checked
against neo4j-agent-memory 0.7.0 (the ``nams`` backend, an async HTTP client):

    MemoryClient(NamsSettings(nams=NamsConfig(...)))        async context manager / connect()
    client.short_term.create_conversation(label, metadata=) -> Conversation (.id is the NAMS id)
    client.short_term.get_conversation(id, limit=)          -> Conversation (.messages)
    client.short_term.bulk_add_messages(id, [{role, content}, ...])
    client.short_term.get_extraction_status(id)             -> ExtractionStatus (.summary counts)
    client.long_term.search_entities(query, limit=)         -> list[Entity]
    client.long_term.wait_for_extraction(session_id=, timeout=) -> bool (never raises on timeout)

Things worth knowing:

* NAMS extracts entities from saved messages in a BACKGROUND pipeline. Saving a message returns
  immediately; the entities it contains become searchable later (usually seconds).
* ``search_entities`` is workspace-wide (all conversations) and nearest-neighbour: it returns the
  top-k closest entities even when none is truly relevant, and it has no relevance score.
* NAMS conversations have no title field. We send the title as conversation ``metadata``.
* The SDK's graph-relationship helpers are not available on NAMS, so this app does not use them.
"""

import asyncio
import logging
from contextlib import contextmanager
from typing import Any

from neo4j_agent_memory import (
    AuthenticationError,
    ConfigurationError,
    ConnectionError as NamsConnectionError,
    MemoryClient,
    MemoryError as NamsError,
    NamsConfig,
    NamsSettings,
    NotFoundError,
    NotSupportedError,
    RateLimitError,
    ValidationError as NamsValidationError,
)
from pydantic import SecretStr

from app.config import Settings
from app.errors import UpstreamError

logger = logging.getLogger(__name__)


# --- error translation ------------------------------------------------------------------------


def to_upstream_error(exc: Exception, action: str) -> UpstreamError:
    """Turn an SDK exception into a safe message + HTTP status (never includes secrets)."""
    if isinstance(exc, AuthenticationError):
        return UpstreamError(
            f"NAMS rejected the credentials while trying to {action}. "
            "Check MEMORY_API_KEY and MEMORY_ENDPOINT.",
            502,
        )
    if isinstance(exc, NotFoundError):
        return UpstreamError(f"NAMS could not find the requested item while trying to {action}.", 404)
    if isinstance(exc, RateLimitError):
        return UpstreamError(f"NAMS rate limit reached while trying to {action}. Retry shortly.", 429)
    if isinstance(exc, NamsConnectionError):  # includes the SDK's TransportError (timeouts, DNS, ...)
        return UpstreamError(f"Could not reach NAMS while trying to {action}.", 503)
    if isinstance(exc, NamsValidationError):
        return UpstreamError(f"NAMS rejected the request while trying to {action}: {str(exc)[:200]}", 422)
    if isinstance(exc, ConfigurationError):
        return UpstreamError(f"NAMS client is misconfigured ({action}): {str(exc)[:200]}", 503)
    if isinstance(exc, NotSupportedError):
        return UpstreamError(f"NAMS does not support this operation ({action}).", 501)
    return UpstreamError(f"NAMS error ({type(exc).__name__}) while trying to {action}.", 502)


@contextmanager
def _translate_errors(action: str):
    """``with _translate_errors("save messages"):`` converts SDK failures to UpstreamError."""
    try:
        yield
    except NamsError as exc:
        logger.warning("NAMS call failed (%s): %s", action, type(exc).__name__)
        raise to_upstream_error(exc, action) from exc
    except asyncio.TimeoutError as exc:
        raise UpstreamError(f"Timed out while trying to {action}.", 503) from exc


# --- helpers ----------------------------------------------------------------------------------


def entity_to_dict(entity: Any) -> dict[str, Any]:
    """Convert an SDK Entity to plain JSON data, keeping whatever fields it actually has.

    Embeddings are dropped (huge and useless to API clients); empty values are omitted so that
    results with different shapes stay readable.
    """
    data = entity.model_dump(mode="json", exclude={"embedding"})
    return {k: v for k, v in data.items() if v not in (None, "", [], {})}


def classify_extraction(summary: dict[str, int]) -> str:
    """Collapse NAMS' per-message extraction counts into one word.

    pending   - at least one message is still queued/being processed
    failed    - nothing pending, but at least one message failed
    completed - nothing pending/failed and at least one message finished
    unknown   - NAMS reported no messages (e.g. status not registered yet): we do NOT claim success
    """
    pending = sum(summary.get(s, 0) for s in ("pending", "processing", "running", "in_progress", "queued"))
    if pending > 0:
        return "pending"
    if summary.get("failed", 0) > 0:
        return "failed"
    if summary.get("completed", 0) > 0:
        return "completed"
    return "unknown"


# --- the service ------------------------------------------------------------------------------


class NamsService:
    """Thin async wrapper around the NAMS SDK.

    The SDK client is created lazily on first use (so the app can start and serve /docs without
    credentials) and is shared by all requests. ``client`` can be injected for tests.
    """

    def __init__(self, settings: Settings, client: Any | None = None):
        self._settings = settings
        self._client = client
        self._lock = asyncio.Lock()

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                config = NamsConfig(
                    endpoint=self._settings.memory_endpoint,
                    api_key=SecretStr(self._settings.require_memory_key()),
                    workspace_id=self._settings.memory_workspace_id,
                    timeout=self._settings.memory_timeout_seconds,
                )
                client = MemoryClient(NamsSettings(nams=config))
                with _translate_errors("connect"):
                    await client.connect()  # makes one authenticated probe request
                self._client = client
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @property
    def is_configured(self) -> bool:
        return self._client is not None or self._settings.memory_api_key is not None

    # --- short-term memory: conversations ---------------------------------------------------

    async def create_conversation(self, title: str) -> str:
        """Create a conversation in NAMS and return the id NAMS generated for it."""
        client = await self._get_client()
        with _translate_errors("create a conversation"):
            # The first argument is only a client-side label; the real id comes back as conv.id.
            conv = await client.short_term.create_conversation(title, metadata={"title": title})
        return str(conv.id)

    async def get_history(self, conversation_id: str, limit: int) -> list[dict[str, str]]:
        """Return the newest ``limit`` user/assistant messages, oldest first.

        Raises UpstreamError(404) if the conversation does not exist.
        """
        client = await self._get_client()
        with _translate_errors("read the conversation"):
            conv = await client.short_term.get_conversation(conversation_id, limit=max(limit, 1))
        turns = []
        for message in conv.messages:
            role = getattr(message.role, "value", message.role)
            if role in ("user", "assistant"):
                turns.append({"role": role, "content": message.content})
        return turns[-limit:] if limit > 0 else []

    async def save_exchange(self, conversation_id: str, user_message: str, assistant_message: str) -> None:
        """Persist one user/assistant turn. This only SAVES; extraction happens later in NAMS."""
        client = await self._get_client()
        with _translate_errors("save the messages"):
            await client.short_term.bulk_add_messages(
                conversation_id,
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ],
            )

    # --- long-term memory: entities ---------------------------------------------------------

    async def search_entities(self, query: str, limit: int) -> list[dict[str, Any]]:
        client = await self._get_client()
        with _translate_errors("search long-term memory"):
            entities = await client.long_term.search_entities(query, limit=limit)
        return [entity_to_dict(e) for e in entities]

    # --- asynchronous extraction ------------------------------------------------------------

    async def extraction_status(self, conversation_id: str) -> dict[str, Any]:
        client = await self._get_client()
        with _translate_errors("read the extraction status"):
            status = await client.short_term.get_extraction_status(conversation_id)
        return {
            "state": classify_extraction(status.summary),
            "summary": status.summary,
            "pending_messages": status.pending_count,
        }

    async def wait_for_extraction(self, conversation_id: str, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for extraction. Returns False on timeout (no exception)."""
        client = await self._get_client()
        with _translate_errors("wait for extraction"):
            return await client.long_term.wait_for_extraction(session_id=conversation_id, timeout=timeout)

    # --- health -----------------------------------------------------------------------------

    async def ping(self) -> None:
        """One cheap authenticated request; raises UpstreamError/ConfigError if NAMS is unusable."""
        client = await self._get_client()
        with _translate_errors("reach NAMS"):
            await client.short_term.list_conversations(limit=1)
