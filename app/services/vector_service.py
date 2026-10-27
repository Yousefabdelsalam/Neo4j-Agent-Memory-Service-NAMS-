"""Document chunking, embeddings and ChromaDB storage/retrieval.

Embeddings are computed here (not by Chroma) so the embedding settings stay separate from the
chat model and Chroma never has to download a local model. Chroma stores whatever vectors we
hand it, using cosine distance (0 = identical, larger = less similar).
"""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import chromadb
import openai

from app.config import Settings
from app.errors import from_openai_error

logger = logging.getLogger(__name__)

EmbedFunction = Callable[[list[str]], list[list[float]]]
EMBED_BATCH_SIZE = 64


@dataclass
class DocumentChunk:
    text: str
    filename: str
    document_id: str
    chunk_index: int
    distance: float


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of at most ``chunk_size`` characters.

    Each chunk ends at the best natural break found in the second half of the window
    (paragraph, line, sentence, then word), and the next chunk starts ``overlap`` characters
    earlier so a sentence cut at the boundary still appears whole in one chunk.
    """
    text = text.replace("\r\n", "\n").strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            for separator in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(separator)
                if cut >= chunk_size // 2:
                    end = start + cut + len(separator)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)  # always move forward
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1  # don't begin the next chunk in the middle of a word
        start = next_start
    return chunks


def make_openai_embedder(settings: Settings) -> EmbedFunction:
    """Build the real embedding function (OpenAI-compatible /embeddings endpoint)."""
    client: openai.OpenAI | None = None

    def embed(texts: list[str]) -> list[list[float]]:
        nonlocal client
        if client is None:  # created on first use so the app starts without credentials
            client = openai.OpenAI(
                api_key=settings.require_embedding_key(),
                base_url=settings.embedding_base_url,
                timeout=60,
            )
        vectors: list[list[float]] = []
        try:
            for i in range(0, len(texts), EMBED_BATCH_SIZE):
                batch = texts[i : i + EMBED_BATCH_SIZE]
                response = client.embeddings.create(model=settings.embedding_model, input=batch)
                vectors.extend(item.embedding for item in response.data)
        except openai.OpenAIError as exc:
            logger.warning("Embedding call failed: %s", type(exc).__name__)
            raise from_openai_error(exc, "create embeddings", "EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL") from exc
        return vectors

    return embed


class VectorService:
    """A persistent ChromaDB collection plus the embedding function that feeds it.

    Methods are synchronous (Chroma and the embedding call block); async callers should use
    ``asyncio.to_thread``.
    """

    def __init__(self, settings: Settings, embed: EmbedFunction | None = None):
        self._settings = settings
        self._embed = embed or make_openai_embedder(settings)
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=None,  # we always pass our own vectors
            configuration={"hnsw": {"space": "cosine"}},
        )

    def count(self) -> int:
        """Number of chunks stored (cheap; used by /health)."""
        return self._collection.count()

    def list_documents(self) -> list[dict]:
        """One entry per indexed document: {document_id, filename, chunks}."""
        metadatas = self._collection.get(include=["metadatas"])["metadatas"]
        documents: dict[str, dict] = {}
        for meta in metadatas:
            entry = documents.setdefault(
                meta["document_id"],
                {"document_id": meta["document_id"], "filename": meta.get("filename", "unknown"), "chunks": 0},
            )
            entry["chunks"] += 1
        return sorted(documents.values(), key=lambda d: d["filename"].lower())

    def index_document(self, text: str, filename: str, document_id: str | None = None) -> dict:
        """Chunk, embed and store a document. Returns {document_id, chunks, replaced}.

        Re-using an existing ``document_id`` replaces that document's chunks instead of adding
        duplicates. Embeddings are computed first, so a failed embedding call leaves the old
        version untouched.
        """
        document_id = document_id or uuid.uuid4().hex
        chunks = chunk_text(text, self._settings.chunk_size, self._settings.chunk_overlap)
        if not chunks:
            raise ValueError("Document has no text to index")

        vectors = self._embed(chunks)

        existing = self._collection.get(where={"document_id": document_id}, include=[])
        replaced = bool(existing["ids"])
        if replaced:
            self._collection.delete(ids=existing["ids"])

        self._collection.add(
            ids=[f"{document_id}:{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=vectors,
            metadatas=[
                {"document_id": document_id, "filename": filename, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )
        return {"document_id": document_id, "chunks": len(chunks), "replaced": replaced}

    def search(self, query: str, top_k: int | None = None) -> list[DocumentChunk]:
        """Return the most similar chunks, closest first, dropping ones that are too distant."""
        top_k = top_k or self._settings.retrieval_top_k
        if self._collection.count() == 0:
            return []
        query_vector = self._embed([query])[0]
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            if distance <= self._settings.retrieval_max_distance:
                hits.append(
                    DocumentChunk(
                        text=text,
                        filename=str(meta.get("filename", "unknown")),
                        document_id=str(meta.get("document_id", "")),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        distance=float(distance),
                    )
                )
        return hits

