import os
import streamlit as st
from streamlit_chat import message
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import shutil
import re
import difflib

# Set GPU acceleration for Ollama
os.environ["OLLAMA_ACCELERATE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Paths
CHROMA_PATH = "chroma_db"
DATA_PATH = "data_brihaspathi"
FORM_PATH = "form.html"  # Path to the quotation form

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

# Load documents and process PDFs
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks):
    if not chunks:
        print("⚠️ No text chunks found. Skipping indexing.")
        return
    
    db = chroma_client
    texts = [chunk.page_content for chunk in chunks]
    
    print(f"📄 Generating embeddings for {len(texts)} chunks...")

    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    # ✅ Fix: Use NumPy's array check to avoid ValueError
    if embeddings is None or np.array(embeddings).size == 0:
        print("❌ ERROR: Embeddings are empty. Check the input data!")
        return  # Prevents inserting empty embeddings

    chunk_ids = [f"{chunk.metadata['source']}:{chunk.metadata['page']}:{i}" for i, chunk in enumerate(chunks)]
    
    db.add_documents(chunks, embeddings=embeddings, ids=chunk_ids)
    print(f"✅ Successfully indexed {len(chunks)} chunks into ChromaDB.")

def index_pdfs():
    print("📄 Loading PDFs...")
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def query_rag(query_text: str):
    # Fixed response for identity-related questions
    identity_questions = [
        "who are you", "what is your name", "introduce yourself", 
        "tell me about yourself", "who am I chatting with"
    ]

    # **Fuzzy Matching for 'Who are you?'**
    best_match = difflib.get_close_matches(query_text.lower().strip(), identity_questions, n=1, cutoff=0.8)
    if best_match:
        return (
            "I am a Support Executive at Brihaspathi Industries Pvt Ltd. "
            "You can reach me at 9676021111, 9676031111, or support@brihaspathi.com. "
            "For technical assistance, please call our Toll-Free number 1800 296 8899."
        )

    # Keywords indicating quotation requests
    quotation_keywords = ["price", "quotation", "cost", "quote", "pricing", "estimate"]
    product_keywords = ["cctv", "solar", "home automation", "sales"]

    # Check if query is about quotations
    if any(q in query_text.lower() for q in quotation_keywords) and any(p in query_text.lower() for p in product_keywords):
        return f"""
        For product quotations, please **<a href="/static/form.html" target="_blank">Click Here</a>** to open the form in a new tab.

        _After submitting, please return to this chat to receive confirmation._
        """

    # Default RAG-based response
    query_embedding = embedding_model.encode([query_text])[0]
    results = chroma_client.similarity_search_with_score(query_text, k=5)
    context_text = "\n".join([doc.page_content for doc, _ in results])
    
    prompt = f"""
    You are a business assistant. Your **only** source of truth is the provided context.

    Context:
    {context_text}

    Answer the question based **strictly** on the context above.
    - If you don't find an answer, **say you don't have enough information** instead of guessing.
    - If the user is asking for quotations, redirect them to the quotation form.
    - Avoid generic definitions or answering beyond the provided data.
    - Provide answers in a direct and informative way.

    Question: {query_text}
    Response:
    """
    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)
    return clean_response(response_text)


def clean_response(response):
    response = re.sub(r'\[.*?\]', '', response)
    unwanted_phrases = ['Dear User,', 'Best regards,', 'Thank you for reaching out', 'Warm regards', 'Sincerely']
    for phrase in unwanted_phrases:
        response = response.replace(phrase, '')
    return response.strip()


# Streamlit UI
st.title("Brihaspathi Chatbot")

# Initialize chat history
if 'message_history' not in st.session_state:
    st.session_state.message_history = []

# User input field
user_input = st.text_input("Ask me anything:")

# Process user input
if st.button("Get Answer") and user_input:
    with st.spinner("Fetching answer..."):
        response = query_rag(user_input)
        st.session_state.message_history.append({"text": user_input, "is_user": True})
        st.session_state.message_history.append({"text": response, "is_user": False})

# Display chat messages
for i, chat_message in enumerate(st.session_state.message_history):
    message(chat_message['text'], is_user=chat_message['is_user'], key=f"chat_{i}")

# Clear chat button
if st.button("Clear Conversation"):
    st.session_state.message_history = []

# ✅ Correcting Form Link in Streamlit UI
if any("quotation" in m["text"].lower() for m in st.session_state.message_history):
    if os.path.exists("static/form.html"):
        st.markdown(f"""
        ### 📝 Get a Quote
        For product quotations, please **<a href="/static/form.html" target="_blank">Click Here</a>** to open the form in a new tab.
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Quotation form not found! Please contact support.")


# ✅ **Show Success Message After Submission**# ✅ **Show Success Message After Submission**
query_params = st.experimental_get_query_params()
if "form_submitted" in query_params:
    st.success("✅ Your request has been submitted! We will get back to you soon.")


if __name__ == "__main__":
    index_pdfs()  # Run PDF indexing before chatbot starts
