def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être positif.")

    if overlap >= chunk_size:
        raise ValueError("overlap doit être inférieur à chunk_size.")

    clean_text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(clean_text):
        end = start + chunk_size
        chunk = clean_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks