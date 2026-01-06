```python
import os
import jwt
from datetime import datetime, timedelta, UTC
from flask import Flask, request, jsonify, g
from dotenv import load_dotenv

from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from helper.jwt_request import jwt_required

"""
Loads environment variables from .env file.
"""
load_dotenv()

"""
GROQ API key.
"""
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

"""
JWT secret key.
"""
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

"""
Pinecone API key.
"""
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

"""
Pinecone index name.
"""
INDEX_NAME = "pdf-rag-index"

"""
Pinecone index.
"""
pc = Pinecone(PINECONE_API_KEY=PINECONE_API_KEY)

"""
Pinecone index stats.
"""
index = pc.Index(INDEX_NAME)
print(index.describe_index_stats())

"""
Creates Pinecone index if it does not exist.
"""
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

"""
Hugging Face embeddings model.
"""
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

"""
Checks if GROQ API key and JWT secret key are set in .env file.
"""
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in .env")

if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY missing in .env")

"""
Configures upload folder and vector database path.
"""
UPLOAD_FOLDER = "uploads"
VECTOR_DB_PATH = "vector_store"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

"""
Creates Flask application.
"""
app = Flask(__name__)

"""
Configures embeddings model.
"""
# EMBEDDINGS (FREE)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

"""
Configures vector database path.
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store")

"""
Loads vector database if it exists.
"""
def load_vector_db():
    """
    Loads vector database from disk if it exists.
    
    Returns:
        FAISS: Loaded vector database.
    """
    global vector_db

    if os.path.exists(VECTOR_DB_PATH):
        try:
            vector_db = FAISS.load_local(
                VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Vector DB loaded from disk")
        except Exception as e:
            print("❌ Failed to load vector DB:", e)

"""
Configures GROQ LLM.
"""
groq_client = Groq(api_key=GROQ_API_KEY)

vector_db = None

"""
Ingests PDF into vector database.
"""
def ingest_pdf(pdf_path):
    """
    Ingests PDF into vector database.
    
    Args:
        pdf_path (str): Path to PDF file.
    """
    global vector_db

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)

    if vector_db is None:
        # First PDF → create new vector DB
        vector_db = FAISS.from_documents(chunks, embeddings)

    else:
        # Additional PDFs → add to existing DB
        vector_db.add_documents(chunks)

    vector_db.save_local(VECTOR_DB_PATH)

"""
Configures login endpoint.
"""
@app.route("/login", methods=["POST"])
def login():
    """
    Handles login request.
    
    Returns:
        jsonify: Login response.
    """
    data = request.get_json()
    email = data.get("email")

    if not email:
        return jsonify({"error": "Email required"}), 400

    payload = {
        "email": email,
        "exp": datetime.now(UTC) + timedelta(hours=1)
    }

    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")

    return jsonify({"access_token": token})

"""
Configures upload PDF endpoint.
"""
@app.route("/upload-pdf", methods=["POST"])
@jwt_required
def upload_pdf():
    """
    Handles upload PDF request.
    
    Returns:
        jsonify: Upload response.
    """
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    ingest_pdf(path)

    return jsonify({"message": "PDF uploaded & indexed successfully"})

"""
Configures chat endpoint.
"""
@app.route("/chat", methods=["POST"])
@jwt_required
def chat():
    """
    Handles chat request.
    
    Returns:
        jsonify: Chat response.
    """
    global vector_db

    if vector_db is None:
        return jsonify({"error": "Vector DB not available"}), 500

    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question required"}), 400

    docs = vector_db.similarity_search(question, k=13)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = completion.choices[0].message.content
    # lines = answer.split("\n")  # Split at each newline

    # save chat history if needed (not implemented here)  
    chat_log_path = os.path.join('chat_logs', 'chat_log.txt')
    os.makedirs('chat_logs', exist_ok=True)  # Create folder if not exists
    with open(chat_log_path, 'a', encoding='utf-8') as f:
        f.write(f"Timestamp:{datetime.now(UTC).isoformat()}\n ")
        f.write(f"User Email:{g.email}\n")
        f.write(f"Question:{question}\n")
        f.write(f"Answer:{answer}\n")
        f.write("_"*50 + "\n")

    return jsonify({
        "question": question,
        "answer": answer
    })

"""
Loads vector database at startup.
"""
load_vector_db()

"""
Runs Flask application.
"""
if __name__ == "__main__":
    app.run(debug=True)
```