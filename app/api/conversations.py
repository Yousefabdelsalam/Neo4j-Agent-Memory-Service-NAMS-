"""Conversations and chat: POST /conversations, POST /chat, GET /conversations/{id}/extraction-status."""

from fastapi import APIRouter, Query, Request

from app.models import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    CreateConversationRequest,
    ExtractionStatusResponse,
    MessagesResponse,
)

router = APIRouter(tags=["conversations and chat"])


@router.post("/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation(body: CreateConversationRequest, request: Request):
    """Create a conversation in NAMS and return the id NAMS generated."""
    conversation_id = await request.app.state.nams.create_conversation(body.title)
    return ConversationResponse(conversation_id=conversation_id, title=body.title)


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    """Run the LangGraph workflow: documents -> memory -> answer -> save."""
    state = await request.app.state.graph.ainvoke(
        {"conversation_id": body.conversation_id, "user_message": body.message}
    )
    memory_status = state["memory_retrieval_status"]
    return ChatResponse(
        answer=state["answer"],
        conversation_id=body.conversation_id,
        document_sources=state.get("document_sources", []),
        memory_retrieval_succeeded=memory_status != "unavailable",
        memory_retrieval_status=memory_status,
        memory_entities_used=state.get("memory_context", []),
        messages_saved=state["message_save_status"] == "saved",
        extraction_status=state["extraction_status"],
        warnings=state.get("warnings", []),
    )


@router.get("/conversations/{conversation_id}/extraction-status", response_model=ExtractionStatusResponse)
async def extraction_status(conversation_id: str, request: Request):
    """Has NAMS finished extracting entities from this conversation's messages yet?"""
    status = await request.app.state.nams.extraction_status(conversation_id)
    return ExtractionStatusResponse(conversation_id=conversation_id, **status)


@router.get("/conversations/{conversation_id}/messages", response_model=MessagesResponse)
async def conversation_messages(conversation_id: str, request: Request, limit: int = Query(default=50, ge=1, le=200)):
    """The newest `limit` saved messages of a conversation, oldest first (read from NAMS)."""
    messages = await request.app.state.nams.get_history(conversation_id, limit)
    return MessagesResponse(conversation_id=conversation_id, messages=messages)
