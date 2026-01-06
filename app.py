import os
import jwt
from datetime import datetime, timedelta, UTC
from flask import Flask, request, jsonify,g
from dotenv import load_dotenv

from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from helper.jwt_request import jwt_required

# ---------------- LOAD ENV ----------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in .env")

if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY missing in .env")

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
VECTOR_DB_PATH = "vector_store"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- EMBEDDINGS (FREE) ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
# ---------------- VECTOR DB PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store")

# ---------------- LOAD VECTOR DB IF EXISTS ----------------
def load_vector_db():
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

# ---------------- GROQ LLM ----------------
groq_client = Groq(api_key=GROQ_API_KEY)

vector_db = None

# ---------------- PDF INGEST ----------------
def ingest_pdf(pdf_path):
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


# ---------------- LOGIN ----------------
@app.route("/login", methods=["POST"])
def login():
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

# ---------------- UPLOAD PDF ----------------
@app.route("/upload-pdf", methods=["POST"])
@jwt_required
def upload_pdf():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    ingest_pdf(path)

    return jsonify({"message": "PDF uploaded & indexed successfully"})

# ---------------- CHAT ----------------
@app.route("/chat", methods=["POST"])
@jwt_required
def chat():
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
    chat_log_path=os.path.join('chat_logs','chat_log.txt')
    os.makedirs('chat_logs',exist_ok=True) # Create folder if not exists
    with open(chat_log_path,'a',encoding='utf-8') as f:
        f.write(f"Timestamp:{datetime.now(UTC).isoformat()}\n ")
        f.write(f"User Email:{g.email}\n")
        f.write(f"Question:{question}\n")
        f.write(f"Answer:{answer}\n")
        f.write("_"*50+"\n")


    return jsonify({
        "question": question,
        "answer": answer
    })
# ---------------- LOAD VECTOR DB AT STARTUP ----------------
load_vector_db()

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
