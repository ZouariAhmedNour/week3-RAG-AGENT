from google.genai import types

from src.config import get_env, get_gemini_client


def embed_text(text: str) -> list[float]:
    client = get_gemini_client()
    model = get_env("EMBEDDING_MODEL", "gemini-embedding-001")

    result = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY"
        ),
    )

    return result.embeddings[0].values