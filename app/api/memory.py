"""GET /memory/search - search NAMS long-term (entity) memory."""

from fastapi import APIRouter, Query, Request

from app.models import MemorySearchResponse

router = APIRouter(prefix="/memory", tags=["memory"])

EMPTY_NOTE = (
    "No entities matched. If you saved messages recently, NAMS may still be extracting entities in the "
    "background - check GET /conversations/{id}/extraction-status and search again."
)
NEAREST_NEIGHBOUR_NOTE = (
    "NAMS entity search is nearest-neighbour: it returns the closest entities it has, "
    "which are not guaranteed to be relevant to the query."
)


@router.get("/search", response_model=MemorySearchResponse)
async def search_memory(
    request: Request,
    q: str = Query(min_length=1, max_length=200, description="Text to search for, e.g. a name.", examples=["Ahmed"]),
    limit: int = Query(default=5, ge=1, le=25),
):
    results = await request.app.state.nams.search_entities(q, limit)
    return MemorySearchResponse(
        query=q,
        count=len(results),
        results=results,
        note=NEAREST_NEIGHBOUR_NOTE if results else EMPTY_NOTE,
    )
