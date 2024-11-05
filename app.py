import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF

# Load the question-answering and embedding models
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

qa_pipeline = load_qa_pipeline()
embedding_model = load_embedding_model()

# Define functions to preprocess and retrieve chunks
def split_text_into_chunks(text, max_length=400):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_faiss_index(chunks):
    chunk_embeddings = embedding_model.encode(chunks)
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings))
    return index, chunk_embeddings

# Set up Streamlit app interface
st.title("Document Question-Answering App")
st.write("Upload a document, ask questions, and get answers based on document context.")

# Upload and process document
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
if uploaded_file is not None:
    # Read PDF content
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text()

    st.write("Document loaded successfully.")

    # Split text into chunks and create an index
    chunks = split_text_into_chunks(full_text)
    index, chunk_embeddings = create_faiss_index(chunks)
    
    st.write("Document indexed for question-answering.")

    # Accept user query and perform retrieval and answering
    query = st.text_input("Ask a question about the document:")
    if query:
        # Embed the query
        query_embedding = embedding_model.encode([query])
        
        # Retrieve top-k most similar chunks
        distances, indices = index.search(query_embedding, 5)
        
        # Combine retrieved chunks
        combined_context = " ".join([chunks[idx] for idx in indices[0]])
        
        # Get answer from QA model
        result = qa_pipeline(question=query, context=combined_context)
        st.write("Answer:", result['answer'])
