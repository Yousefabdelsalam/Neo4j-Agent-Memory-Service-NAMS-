"""FastAPI application: wires the services, the LangGraph workflow and the routers together.

Run with:  uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import conversations, documents, memory
from app.config import Settings
from app.errors import ConfigError, UpstreamError
from app.graph.workflow import build_chat_graph
from app.models import HealthResponse
from app.services.llm_service import LLMService
from app.services.nams_service import NamsService
from app.services.vector_service import VectorService

STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DESCRIPTION = """
Minimal demo of **NAMS** (long-term memory) + **ChromaDB** (document retrieval) + **LangGraph** (workflow).

Try it in order: `POST /documents/upload` -> `POST /conversations` -> `POST /chat` -> `GET /memory/search`.
"""


def create_app(
    settings: Settings | None = None,
    vector: VectorService | None = None,
    nams: NamsService | None = None,
    llm: LLMService | None = None,
) -> FastAPI:
    """Build the app. The optional arguments let tests supply their own service instances."""
    settings = settings or Settings()
    vector = vector or VectorService(settings)
    nams = nams or NamsService(settings)
    llm = llm or LLMService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await nams.aclose()  # close the NAMS HTTP session, if one was opened

    app = FastAPI(title="MemoryRAG", description=DESCRIPTION, version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.vector = vector
    app.state.nams = nams
    app.state.llm = llm
    app.state.graph = build_chat_graph(vector, nams, llm, settings)

    app.include_router(documents.router)
    app.include_router(conversations.router)
    app.include_router(memory.router)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def web_ui():
        return FileResponse(STATIC_DIR / "index.html")

    @app.exception_handler(ConfigError)
    async def config_error_handler(request: Request, exc: ConfigError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(UpstreamError)
    async def upstream_error_handler(request: Request, exc: UpstreamError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health(deep: bool = False):
        """Basic health. Never exposes secrets and makes no LLM calls.

        With `deep=true` it also makes ONE cheap authenticated request to NAMS.
        """
        try:
            chroma = {"status": "ok", "chunks_indexed": vector.count()}
        except Exception as exc:  # report, don't crash the health check
            chroma = {"status": "error", "error": type(exc).__name__}

        nams_info = {"configured": nams.is_configured}
        if deep and nams.is_configured:
            try:
                await nams.ping()
                nams_info["reachable"] = True
            except (UpstreamError, ConfigError) as exc:
                nams_info["reachable"] = False
                nams_info["error"] = getattr(exc, "message", str(exc))

        llm_info = {"configured": llm.is_configured, "provider": settings.llm_provider, "model": settings.llm_model}

        degraded = (
            chroma["status"] != "ok"
            or not nams_info["configured"]
            or nams_info.get("reachable") is False
            or not llm_info["configured"]
        )
        return HealthResponse(status="degraded" if degraded else "ok", chroma=chroma, nams=nams_info, llm=llm_info)

    return app


app = create_app()
