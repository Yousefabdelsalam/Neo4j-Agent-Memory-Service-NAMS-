"""The chat workflow: four nodes that always run in the same order.

    START -> retrieve_document_context -> retrieve_memory_context -> generate_answer
          -> save_conversation -> END

No branching, no loops, no checkpointer. Each node receives the current state and returns only
the keys it wants to update. The services are passed in (not imported as globals), so the graph
is easy to test with stand-ins.
"""

import asyncio
import logging

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.errors import UpstreamError
from app.graph.state import ChatState
from app.services.llm_service import LLMService
from app.services.nams_service import NamsService
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


def build_chat_graph(vector: VectorService, nams: NamsService, llm: LLMService, settings: Settings):
    """Create and compile the chat graph. Call ``await graph.ainvoke({...})`` to run it."""

    async def retrieve_document_context(state: ChatState) -> dict:
        """Search ChromaDB for chunks related to the user's message."""
        # Chroma and the embedding call are blocking, so run them off the event loop.
        chunks = await asyncio.to_thread(vector.search, state["user_message"])
        sources = list(dict.fromkeys(chunk.filename for chunk in chunks))  # unique, in order
        return {"document_context": chunks, "document_sources": sources}

    async def retrieve_memory_context(state: ChatState) -> dict:
        """Load this conversation's recent messages and search NAMS long-term memory.

        Failures here are not fatal: the answer is still generated, with
        ``memory_retrieval_status = "unavailable"``. A conversation that does not exist (404) is
        the exception - there is nothing sensible to answer, so that error propagates.
        """
        update: dict = {"history": [], "memory_context": [], "warnings": []}

        try:
            update["history"] = await nams.get_history(state["conversation_id"], settings.history_messages)
        except UpstreamError as exc:
            if exc.status_code == 404:
                raise UpstreamError(
                    "Conversation not found in NAMS. Create one with POST /conversations first.", 404
                ) from exc
            update["warnings"].append(f"Conversation history unavailable: {exc.message}")

        try:
            entities = await nams.search_entities(state["user_message"], settings.memory_entity_limit)
            update["memory_context"] = entities
            update["memory_retrieval_status"] = "ok" if entities else "empty"
        except UpstreamError as exc:
            update["memory_retrieval_status"] = "unavailable"
            update["warnings"].append(f"Long-term memory unavailable: {exc.message}")
        return update

    async def generate_answer(state: ChatState) -> dict:
        """Ask the LLM, giving it the document excerpts, memory entities and recent messages."""
        answer = await llm.answer(
            user_message=state["user_message"],
            history=state.get("history", []),
            chunks=state.get("document_context", []),
            entities=state.get("memory_context", []),
        )
        return {"answer": answer}

    async def save_conversation(state: ChatState) -> dict:
        """Persist the exchange in NAMS, then report (never wait forever for) extraction.

        Two separate facts are reported:
        - message_save_status: did NAMS store the messages?
        - extraction_status:   has NAMS finished turning them into long-term entities?
        Saving can succeed while extraction is still "pending" - that is normal.
        """
        try:
            await nams.save_exchange(state["conversation_id"], state["user_message"], state["answer"])
        except UpstreamError as exc:
            return {
                "message_save_status": "failed",
                "extraction_status": "not_checked",
                "warnings": [f"Messages were NOT saved: {exc.message}"],
            }

        update: dict = {"message_save_status": "saved", "extraction_status": "unknown", "warnings": []}
        try:
            if settings.extraction_wait_seconds > 0:
                await nams.wait_for_extraction(state["conversation_id"], settings.extraction_wait_seconds)
            status = await nams.extraction_status(state["conversation_id"])
            update["extraction_status"] = status["state"]
        except UpstreamError as exc:
            update["warnings"].append(f"Could not read extraction status: {exc.message}")
        return update

    graph = StateGraph(ChatState)
    graph.add_node("retrieve_document_context", retrieve_document_context)
    graph.add_node("retrieve_memory_context", retrieve_memory_context)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("save_conversation", save_conversation)

    graph.add_edge(START, "retrieve_document_context")
    graph.add_edge("retrieve_document_context", "retrieve_memory_context")
    graph.add_edge("retrieve_memory_context", "generate_answer")
    graph.add_edge("generate_answer", "save_conversation")
    graph.add_edge("save_conversation", END)
    return graph.compile()
