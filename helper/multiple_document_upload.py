from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
)

# -----------------------------
# Supported file extensions
# -----------------------------
SUPPORTED_EXTENSIONS = {
    "pdf", "txt", "md", "csv",
    "doc", "docx",
    "xls", "xlsx",
    "ppt", "pptx",
    "html", "htm"
}


# -----------------------------
# Loader factory
# -----------------------------
def get_loader(file_path: str, ext: str):
    if ext == "pdf":
        return PyPDFLoader(file_path)

    if ext in ["txt", "md"]:
        return TextLoader(file_path, encoding="utf-8")

    if ext == "csv":
        return CSVLoader(
            file_path,
            encoding="utf-8",
            csv_args={"delimiter": ","}
        )

    if ext in ["doc", "docx"]:
        return UnstructuredWordDocumentLoader(file_path)

    if ext in ["xls", "xlsx"]:
        return UnstructuredExcelLoader(file_path)

    if ext in ["ppt", "pptx"]:
        return UnstructuredPowerPointLoader(file_path)

    if ext in ["html", "htm"]:
        return UnstructuredHTMLLoader(file_path)

    return None


# -----------------------------
# Core ingest function
# -----------------------------
def ingest_document(
    *,
    file_path: str,
    filename: str,
    user_email: str,
    vector_db,
    embeddings,
    persist_directory: str,
):
    ext = filename.lower().split(".")[-1]

    # ---- validate extension ----
    if ext not in SUPPORTED_EXTENSIONS:
        return {
            "uploaded": True,
            "indexed": False,
            "filename": filename,
            "reason": "Unsupported file type",
            "vector_db": vector_db,
        }

    loader = get_loader(file_path, ext)
    if not loader:
        return {
            "uploaded": True,
            "indexed": False,
            "filename": filename,
            "reason": "No loader available for this file type",
            "vector_db": vector_db,
        }

    # ---- load document safely ----
    try:
        docs = loader.load()
    except Exception as e:
        return {
            "uploaded": True,
            "indexed": False,
            "filename": filename,
            "reason": f"Loader error: {str(e)}",
            "vector_db": vector_db,
        }

    if not docs:
        return {
            "uploaded": True,
            "indexed": False,
            "filename": filename,
            "reason": "No readable content found",
            "vector_db": vector_db,
        }

    # ---- split text ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        return {
            "uploaded": True,
            "indexed": False,
            "filename": filename,
            "reason": "No valid text chunks (scanned / image-based / text-box document)",
            "vector_db": vector_db,
        }

    # ---- metadata ----
    for chunk in chunks:
        chunk.metadata.update({
            "source": filename.lower(),
            "user": user_email,
            "type": ext
        })

    # ---- vector store ----
    if vector_db is None:
        vector_db = Chroma(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    else:
        vector_db.add_documents(chunks)

    return {
        "uploaded": True,
        "indexed": True,
        "filename": filename,
        "chunks": len(chunks),
        "vector_db": vector_db,
    }
