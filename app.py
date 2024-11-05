import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF

# Load the question-answering and embedding models with caching for efficiency
@st.cache_resource
def load_qa_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-large")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="google/flan-t5-large")

qa_pipeline = load_qa_pipeline()
embedding_model = load_embedding_model()
summarizer = load_summarizer()

# Function to split text into smaller chunks
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

# Function to create FAISS index for the document chunks
def create_faiss_index(chunks):
    chunk_embeddings = embedding_model.encode(chunks)
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings))
    return index, chunk_embeddings

# Function to retrieve top-k chunks and truncate if necessary
def retrieve_and_truncate_chunks(query, chunks, index, top_k=5, max_length=512):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    combined_chunks = " ".join([chunks[idx] for idx in indices[0]])
    
    # Truncate the combined text to fit within the model’s max token limit
    return ' '.join(combined_chunks.split()[:max_length])

# Set up Streamlit app interface
st.title("Enhanced Document Question-Answering and Summarization App")
st.write("Upload a document, ask questions, and get detailed answers with a document summary.")

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
    
    # Generate a summary of the document
    document_summary = summarizer(full_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    st.subheader("Document Summary")
    st.write(document_summary)

    # Accept user query and perform retrieval and answering
    query = st.text_input("Ask a question about the document:")
    if query:
        # Retrieve top-k relevant chunks based on semantic similarity and truncate
        relevant_text = retrieve_and_truncate_chunks(query, chunks, index, top_k=5, max_length=512)
        
        # Generate answer using the retrieved relevant context
        prompt = f"Context: {relevant_text}\n\nQuestion: {query}\nAnswer:"
        answer = qa_pipeline(prompt, max_length=150, do_sample=True)[0]['generated_text']
        
        st.subheader("Answer")
        st.write(answer)
        
        st.subheader("Document Summary")
        st.write(document_summary)
