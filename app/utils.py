from langchain_community.document_loaders import PyPDFLoader
import requests
from pathlib import Path

def load_pdf(path_or_url: str):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        response = requests.get(path_or_url)
        response.raise_for_status()
        file_path = Path("temp.pdf")
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        file_path = Path(path_or_url)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(str(file_path))
    return loader.load()  # ✅ Returns list of Document objects
