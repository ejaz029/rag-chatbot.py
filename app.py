import os
from dotenv import load_dotenv
import groq
from sentence_transformers import SentenceTransformer
import chromadb
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Increase timeout to 300 seconds
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

# Initialize API and database clients
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Error: GROQ_API_KEY not found in .env file.")
    st.stop()

groq_client = groq.Client(api_key=groq_api_key)
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get the ChromaDB collection
collection_name = "documents"
try:
    collection = chroma_client.get_collection(name=collection_name)
except Exception:
    collection = chroma_client.create_collection(name=collection_name)

# Load documents from file
file_path = 'documents.txt'
try:
    with open(file_path, 'r') as file:
        documents = file.readlines()
except FileNotFoundError:
    st.error(f"Error: File '{file_path}' not found.")
    st.stop()

# Add documents to the collection if they don't already exist
existing_ids = set(collection.get()["ids"])
new_documents, new_embeddings, new_ids = [], [], []

for idx, doc in enumerate(documents):
    doc_id = str(idx)
    if doc_id not in existing_ids:
        new_documents.append(doc)
        new_embeddings.append(embedding_model.encode(doc).tolist())
        new_ids.append(doc_id)

if new_documents:
    collection.add(documents=new_documents, embeddings=new_embeddings, ids=new_ids)

# Define the RAG chatbot function
def rag_chatbot(query):
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=2)
    context = " ".join(results['documents'][0]) if results["documents"] else ""
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="mixtral-8x7b-32768",
            max_tokens=50
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit UI
st.title("AI Chatbot")
st.write("Ask me anything based on the uploaded documents!")

query = st.text_input("You: ", "")
if st.button("Ask") and query:
    response = rag_chatbot(query)
    st.text_area("Bot:", value=response, height=100, max_chars=None)
