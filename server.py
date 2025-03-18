from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import difflib

# Initialize FastAPI
app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set GPU acceleration for Ollama
os.environ["OLLAMA_ACCELERATE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Paths
CHROMA_PATH = "chroma_db"
DATA_PATH = "data_brihaspathi"

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class SentenceTransformerEmbedding:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

def get_embedding_function():
    return SentenceTransformerEmbedding()

# Initialize ChromaDB
embedding_function = get_embedding_function()
chroma_client = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Class for user query
class UserQuery(BaseModel):
    message: str

# Function to process chatbot response
def query_rag(query_text: str):
    identity_questions = ["who are you", "what is your name", "introduce yourself", "tell me about yourself", "who am I chatting with"]
    best_match = difflib.get_close_matches(query_text.lower().strip(), identity_questions, n=1, cutoff=0.8)

    if best_match:
        return {
            "response": (
                "I am a Support Executive at Brihaspathi Industries Pvt Ltd. "
                "You can reach me at 9676021111, 9676031111, or support@brihaspathi.com. "
                "For technical assistance, please call our Toll-Free number 1800 296 8899."
            ),
            "show_form": False
        }

    quotation_keywords = ["price", "quotation", "cost", "quote", "pricing", "estimate", "Quotation", "Quote"]
    product_keywords = ["cctv", "solar", "home automation", "sales"]

    if any(q in query_text.lower() for q in quotation_keywords) or any(p in query_text.lower() for p in product_keywords):
        return {
            "response": "Please fill in the form below to get a quotation.",
            "show_form": True
        }

    query_embedding = embedding_model.encode([query_text])[0]
    results = chroma_client.similarity_search_with_score(query_text, k=5)

# 🛑 If no relevant context, return a direct response
    if not results:
        return {
            "response": "I don’t have enough information to answer that. Please contact us directly for further assistance at 9676021111, 9676031111, or support@brihaspathi.com.",
            "show_form": False
        }

    context_text = "\n".join([doc.page_content for doc, _ in results])


    prompt = f"""
    You are a business assistant and company chatbot. Your **only** source of truth is the provided context **never** use placeholders such as [Insert Quotation Form] or general templates. Always provide specific and actionable responses based on the given query. If the user requests a quotation, Please go through guidelines and respond as per guidelines. Do not deviate from this instruction.

    Context:
    {context_text}

    Answer the question based **strictly** on the context above.
    - If you don't find an answer, **say you don't have enough information** instead of guessing.
    - If the user is asking for quotations, redirect them to the quotation form.
    - Avoid generic definitions or answering beyond the provided data.
    - Provide answers in a direct and informative way and use bullet points if there is a list of items.
    - **Do not generate responses with placeholders such as [insert link], [company contact details], or [your name]. Instead, directly give the quotation form instead of any fake links or fake link placeholders.**
    - **Prevent hallucination by only responding with factual, verified information from the given data. Do not invent details**
    - **DO NOT make up company names or information. Only refer to Brihaspathi Technologies Limited.**
    - Provide a direct and professional response. Do not include phrases like "According to the provided context" or "Based on the given data."


    Question: {query_text}
    Response:
    """
    model = Ollama(model="llama3.2",temperature=0.2, top_k=40)
    response_text = model.invoke(prompt).strip()

    # 🛑 Remove redundant phrases
    unwanted_phrases = [
        "According to the provided context,", 
        "According to context,", 
        "Comptech",
        "vasavi",
        "Based on the given data,"
    ]

    for phrase in unwanted_phrases:
        response_text = response_text.replace(phrase, "").strip()

    return {"response": response_text, "show_form": False}

# API endpoint to handle user messages
@app.post("/chat")
async def chat(query: UserQuery):
    return query_rag(query.message)

# API endpoint for form submission
@app.post("/submit_form")
async def submit_form(name: str = Form(...), email: str = Form(...), phone: str = Form(...), interest: str = Form(...)):
    return {"message": "✅ Your request has been submitted! We will get back to you soon."}

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
