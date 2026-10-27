"""POST /documents/upload - index a .txt, .md or .pdf file into ChromaDB."""

import io
import re
from pathlib import PurePath

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pypdf import PdfReader
from pypdf.errors import PyPdfError
from starlette.concurrency import run_in_threadpool

from app.models import DocumentInfo, UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def extract_pdf_text(raw: bytes) -> str:
    """Return the text layer of a PDF. Raises ValueError with a user-friendly message on failure.

    Only text PDFs work: a scanned PDF (pictures of pages) has no text layer and would need OCR.
    """
    if not raw.lstrip().startswith(b"%PDF-"):
        raise ValueError("The file is not a valid PDF.")
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("The PDF is password-protected.")
        pages = [page.extract_text() or "" for page in reader.pages]
    except PyPdfError:
        raise ValueError("The PDF could not be read (it may be corrupted).") from None
    text = "\n\n".join(page.strip() for page in pages if page.strip())
    if not text:
        raise ValueError("No text found in the PDF. Scanned/image-only PDFs are not supported (they need OCR).")
    return text


@router.get("", response_model=list[DocumentInfo])
async def list_documents(request: Request):
    """List the documents currently indexed in ChromaDB."""
    return await run_in_threadpool(request.app.state.vector.list_documents)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="A .txt, .md or .pdf file."),
    document_id: str | None = Form(
        default=None,
        description="Optional. Re-using an existing id replaces that document instead of duplicating it.",
    ),
):
    settings = request.app.state.settings
    vector = request.app.state.vector

    filename = PurePath(file.filename or "").name  # drop any directory part
    if PurePath(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Only .txt, .md and .pdf files are supported.")
    if document_id is not None and not DOCUMENT_ID_PATTERN.match(document_id):
        raise HTTPException(422, "document_id may only contain letters, digits, '.', '_' and '-' (max 64).")

    limit = settings.max_upload_bytes
    raw = await file.read(limit + 1)  # read one byte past the limit to detect oversize files
    if len(raw) > limit:
        raise HTTPException(413, f"File is larger than the {settings.max_upload_mb:g} MB limit.")
    if not raw.strip():
        raise HTTPException(400, "The file is empty.")

    if filename.lower().endswith(".pdf"):
        try:
            text = await run_in_threadpool(extract_pdf_text, raw)  # parsing is CPU work
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(400, "The file is not valid UTF-8 text.") from None
        if "\x00" in text:
            raise HTTPException(400, "The file looks binary, not text.")
        if not text.strip():
            raise HTTPException(400, "The file contains only whitespace.")

    # Embedding + Chroma writes block, so run them in a worker thread.
    result = await run_in_threadpool(vector.index_document, text, filename, document_id)
    return UploadResponse(
        document_id=result["document_id"],
        filename=filename,
        chunks_indexed=result["chunks"],
        replaced_existing=result["replaced"],
    )
