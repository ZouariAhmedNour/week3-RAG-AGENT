import argparse
import hashlib

import chromadb

from src.chunking import chunk_text
from src.config import get_env
from src.embeddings import embed_text
from src.loaders import load_documents


def make_chunk_id(source: str, chunk_index: int, chunk: str) -> str:
    raw = f"{source}-{chunk_index}-{chunk[:50]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def get_collection(reset: bool = False):
    chroma_path = get_env("CHROMA_PATH", "chroma_db")
    collection_name = get_env("COLLECTION_NAME", "course_docs")

    client = chromadb.PersistentClient(path=chroma_path)

    if reset:
        try:
            client.delete_collection(name=collection_name)
            print(f"Collection supprimée : {collection_name}")
        except Exception:
            pass

    return client.get_or_create_collection(name=collection_name)


def index_documents(reset: bool = False) -> None:
    documents = load_documents("data/docs")
    collection = get_collection(reset=reset)

    total_chunks = 0

    for document in documents:
        source = document["source"]
        text = document["text"]
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(source, index, chunk)
            embedding = embed_text(chunk)

            collection.add(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[
                    {
                        "source": source,
                        "chunk_index": index,
                    }
                ],
            )

            total_chunks += 1
            print(f"[INDEXED] {source} - chunk {index}")

    print(f"\nIndexation terminée : {total_chunks} chunks ajoutés.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indexation des documents dans ChromaDB."
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Supprimer l'ancienne collection avant indexation.",
    )

    args = parser.parse_args()
    index_documents(reset=args.reset)


if __name__ == "__main__":
    main()