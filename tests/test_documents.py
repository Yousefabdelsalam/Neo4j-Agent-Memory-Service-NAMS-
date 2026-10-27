"""Feature A: chunking, upload validation and ChromaDB indexing/retrieval."""

import pytest

from app.services.vector_service import chunk_text


# --- chunking ---------------------------------------------------------------------------------


def test_chunks_respect_size_and_cover_the_text():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0].startswith("word0") and chunks[-1].endswith("word299")


def test_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=60, overlap=15)
    # the first words of each chunk already appeared at the end of the previous one
    first_word_of_next = chunks[1].split()[0]
    assert first_word_of_next in chunks[0].split()


def test_unbroken_text_still_terminates():
    chunks = chunk_text("x" * 1000, chunk_size=100, overlap=10)
    assert len(chunks) >= 10 and all(len(c) <= 100 for c in chunks)


def test_blank_text_gives_no_chunks():
    assert chunk_text("  \n\n ", 100, 10) == []


# --- upload validation ------------------------------------------------------------------------


def upload(client, name="notes.txt", content=b"hello world", **data):
    return client.post("/documents/upload", files={"file": (name, content, "text/plain")}, data=data)


def test_rejects_unsupported_extension(client):
    assert upload(client, "report.docx", b"PK").status_code == 415


def test_rejects_empty_file(client):
    assert upload(client, content=b"").status_code == 400


def test_rejects_whitespace_only_file(client):
    assert upload(client, content=b"  \n \n").status_code == 400


def test_rejects_oversized_file(client):
    assert upload(client, content=b"a" * 20_000).status_code == 413


def test_rejects_non_utf8_file(client):
    assert upload(client, content=b"\xff\xfe\x00bad").status_code == 400


def test_rejects_bad_document_id(client):
    assert upload(client, document_id="../evil").status_code == 422


# --- indexing ---------------------------------------------------------------------------------

ATLAS = "Project Atlas is a satellite imaging platform built by the geo team. " * 8
RECIPES = "Grandma's lasagna uses fresh basil, ricotta and slow cooked tomato sauce. " * 8


def test_upload_indexes_chunks_with_metadata(client, vector):
    response = upload(client, "atlas.md", ATLAS.encode())
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "atlas.md"
    assert body["chunks_indexed"] > 1
    assert body["document_id"] and body["replaced_existing"] is False
    assert vector.count() == body["chunks_indexed"]


def test_directory_part_of_filename_is_dropped(client):
    assert upload(client, "../../etc/atlas.txt", b"some text").json()["filename"] == "atlas.txt"


def test_reusing_document_id_replaces_instead_of_duplicating(client, vector):
    first = upload(client, "atlas.md", ATLAS.encode(), document_id="atlas").json()
    second = upload(client, "atlas.md", b"Short replacement text.", document_id="atlas").json()
    assert second["replaced_existing"] is True and second["document_id"] == "atlas"
    assert second["chunks_indexed"] == 1 < first["chunks_indexed"]
    assert vector.count() == 1  # old chunks are gone


def test_search_returns_the_relevant_document(client, vector):
    upload(client, "atlas.md", ATLAS.encode())
    upload(client, "recipes.txt", RECIPES.encode())
    hits = vector.search("satellite imaging platform", top_k=3)
    assert hits and hits[0].filename == "atlas.md"
    assert hits[0].distance <= hits[-1].distance


def test_search_on_empty_collection_returns_nothing(vector):
    assert vector.search("anything") == []


def test_failed_embedding_keeps_the_previous_version(settings):
    from app.errors import UpstreamError
    from app.services.vector_service import VectorService
    from tests.conftest import fake_embed

    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] > 1:
            raise UpstreamError("embedding down", 503)
        return fake_embed(texts)

    service = VectorService(settings, embed=flaky)
    service.index_document("original text", "a.txt", document_id="doc")
    with pytest.raises(UpstreamError):
        service.index_document("new text", "a.txt", document_id="doc")
    assert service.count() == 1


# --- PDF uploads ------------------------------------------------------------------------------


def make_pdf(text: str) -> bytes:
    """Build a tiny valid single-page PDF whose text layer contains ``text`` (test helper)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref_at)
    return out


def pdf_bytes(writer) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_pdf_text_is_indexed_and_searchable(client, vector):
    pdf = make_pdf("Project Atlas is a satellite imaging platform for flood monitoring")
    response = upload(client, "atlas.pdf", pdf)
    assert response.status_code == 200
    assert response.json()["filename"] == "atlas.pdf" and response.json()["chunks_indexed"] >= 1
    hits = vector.search("satellite imaging flood", top_k=2)
    assert hits and hits[0].filename == "atlas.pdf" and "satellite" in hits[0].text


def test_pdf_without_text_layer_is_rejected_clearly(client):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # like a scanned page: no text
    response = upload(client, "scan.pdf", pdf_bytes(writer))
    assert response.status_code == 400 and "OCR" in response.json()["detail"]


def test_password_protected_pdf_is_rejected(client):
    from io import BytesIO

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(BytesIO(make_pdf("secret text"))))
    writer.encrypt("pw")
    response = upload(client, "locked.pdf", pdf_bytes(writer))
    assert response.status_code == 400 and "password" in response.json()["detail"]


def test_fake_and_corrupted_pdfs_are_rejected(client):
    assert upload(client, "fake.pdf", b"just some text").status_code == 400
    assert upload(client, "broken.pdf", b"%PDF-1.4\n1 0 obj << /Broken").status_code == 400
