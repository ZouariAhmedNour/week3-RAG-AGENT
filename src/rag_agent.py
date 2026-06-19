import argparse
import chromadb

from src.config import get_env, get_gemini_client
from src.embeddings import embed_text


SYSTEM_PROMPT = """
Tu es un assistant RAG spécialisé dans la réponse à partir de documents.

Règles obligatoires :
- Réponds uniquement à partir du contexte fourni.
- Si le contexte ne contient pas la réponse, dis clairement :
  "Je ne trouve pas cette information dans les documents fournis."
- Ne pas inventer.
- Réponds en français.
- Cite les sources utilisées à la fin.
"""


def get_collection():
    chroma_path = get_env("CHROMA_PATH", "chroma_db")
    collection_name = get_env("COLLECTION_NAME", "course_docs")

    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_collection(name=collection_name)


def retrieve_context(question: str, top_k: int = 3) -> list[dict]:
    collection = get_collection()
    question_embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    contexts = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, metadata, distance in zip(documents, metadatas, distances):
        contexts.append(
            {
                "text": doc,
                "source": metadata.get("source"),
                "chunk_index": metadata.get("chunk_index"),
                "distance": distance,
            }
        )

    return contexts


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_text = ""

    for i, item in enumerate(contexts, start=1):
        context_text += f"""
[Source {i}]
Fichier : {item["source"]}
Chunk : {item["chunk_index"]}
Contenu :
{item["text"]}
"""

    return f"""
{SYSTEM_PROMPT}

Contexte récupéré :
{context_text}

Question utilisateur :
{question}

Réponse :
"""


def ask_rag(question: str, top_k: int = 3) -> str:
    client = get_gemini_client()
    model = get_env("GEMINI_MODEL", "gemini-2.5-flash")

    contexts = retrieve_context(question, top_k=top_k)
    prompt = build_prompt(question, contexts)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    sources = "\n".join(
        {
            f"- {item['source']} / chunk {item['chunk_index']}"
            for item in contexts
        }
    )

    return f"{response.text}\n\nSources récupérées :\n{sources}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent RAG qui répond à partir des documents indexés."
    )

    parser.add_argument(
        "--ask",
        type=str,
        required=True,
        help="Question à poser à tes documents.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de chunks à récupérer.",
    )

    args = parser.parse_args()

    try:
        answer = ask_rag(args.ask, top_k=args.top_k)
        print("\n--- Réponse RAG ---")
        print(answer)

    except Exception as error:
        print(f"Erreur : {error}")


if __name__ == "__main__":
    main()