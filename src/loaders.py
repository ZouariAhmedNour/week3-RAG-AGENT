from pathlib import Path
from pypdf import PdfReader


def load_txt_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages_text = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(f"\n[Page {page_index}]\n{text.strip()}")

    return "\n".join(pages_text).strip()


def load_documents(folder_path: str = "data/docs") -> list[dict]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable : {folder_path}")

    documents = []

    for path in folder.rglob("*"):
        if path.is_dir():
            continue

        if path.suffix.lower() == ".txt":
            text = load_txt_file(path)
        elif path.suffix.lower() == ".pdf":
            text = load_pdf_file(path)
        else:
            continue

        if text:
            documents.append(
                {
                    "source": str(path),
                    "text": text,
                }
            )

    if not documents:
        raise ValueError(
            "Aucun document .txt ou .pdf trouvé dans data/docs."
        )

    return documents