import pytest

from src.chunking import chunk_text


def test_chunk_text_returns_chunks():
    text = "A" * 2000
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_chunk_text_small_text():
    text = "Petit texte."
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    assert chunks == ["Petit texte."]


def test_chunk_text_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("test", chunk_size=0, overlap=0)


def test_chunk_text_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("test", chunk_size=100, overlap=100)