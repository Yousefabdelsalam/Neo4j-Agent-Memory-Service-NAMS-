"""The data that flows through the chat graph.

This is LangGraph's SHORT-LIVED state: it exists for the duration of one /chat request and is then
thrown away. Anything worth remembering is written to NAMS by the last node; this state is not a
memory store and is never checkpointed.
"""

import operator
from typing import Annotated, TypedDict

from app.services.vector_service import DocumentChunk


class ChatState(TypedDict, total=False):
    # inputs
    conversation_id: str
    user_message: str

    # filled by retrieve_document_context
    document_context: list[DocumentChunk]
    document_sources: list[str]

    # filled by retrieve_memory_context
    history: list[dict[str, str]]        # earlier messages of this conversation (from NAMS)
    memory_context: list[dict]           # entities from NAMS long-term memory
    memory_retrieval_status: str         # "ok" | "empty" | "unavailable"

    # filled by generate_answer
    answer: str

    # filled by save_conversation
    message_save_status: str             # "saved" | "failed"
    extraction_status: str               # "pending" | "completed" | "failed" | "unknown" | "not_checked"

    # nodes append here; the reducer concatenates lists instead of overwriting
    warnings: Annotated[list[str], operator.add]
