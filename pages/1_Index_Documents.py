import streamlit as st
import markitdown
from beaver import BeaverDB, Document
from fastembed import TextEmbedding
import re
from typing import List

# --- Configuration ---
DB_PATH = "knowledge_base.db"
COLLECTION_NAME = "documents"
MIN_CHUNK_LENGTH = 1000
MAX_CHUNK_LENGTH = 2000

# --- Helper Functions ---

@st.cache_resource
def get_db():
    """Initializes and returns the BeaverDB instance."""
    return BeaverDB(DB_PATH)

@st.cache_resource
def get_embedding_model():
    """Loads and caches the fastembed model."""
    # Using 'BAAI/bge-small-en-v1.5' - a small and efficient model
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def chunk_text(text: str, min_len: int, max_len: int) -> List[str]:
    """
    Chunks text by paragraphs, merging small paragraphs to meet length constraints.
    """
    paragraphs = re.split(r'\n{2,}', text) # Split by two or more newlines

    chunks = []
    current_chunk = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        # If adding the next paragraph fits within the max length, add it
        if len(current_chunk) + len(p) + 1 <= max_len:
            current_chunk += (" " + p if current_chunk else p)
        # If the current chunk is already long enough, bank it and start a new one
        elif len(current_chunk) >= min_len:
            chunks.append(current_chunk)
            current_chunk = p
        # Otherwise, the current chunk is too short, so just add the new paragraph to it
        else:
             current_chunk += (" " + p)

    # Add the last remaining chunk if it's valid
    if current_chunk and len(current_chunk) >= min_len:
        chunks.append(current_chunk)

    return chunks

# --- Streamlit UI ---

st.set_page_config(page_title="Index Documents", page_icon="📄")
st.title("📄 Index Documents")
st.markdown("Upload your `.docx` or `.pdf` files here. The content will be extracted, chunked, and indexed into the BeaverDB knowledge base.")

db = get_db()
embedding_model = get_embedding_model()

uploaded_files = st.file_uploader(
    "Choose your knowledge base files",
    accept_multiple_files=True,
    type=['pdf', 'docx']
)

if st.button("Index Uploaded Files", type="primary"):
    if uploaded_files:
        total_chunks_indexed = 0
        docs_collection = db.collection(COLLECTION_NAME)

        st.info("Starting the indexing process... Please wait.")

        overall_progress = st.progress(0, text="Initializing...")

        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            update_text = f"Processing file {i+1}/{len(uploaded_files)}: {file_name}"
            overall_progress.progress(i / len(uploaded_files), text=update_text)

            try:
                # 1. Extract text with markitdown
                markdown_text = markitdown.MarkItDown().convert(uploaded_file).markdown

                # 2. Chunk the text
                chunks = chunk_text(markdown_text, MIN_CHUNK_LENGTH, MAX_CHUNK_LENGTH)

                if not chunks:
                    st.warning(f"Could not extract any suitable text chunks from '{file_name}'. Skipping.")
                    continue

                # 3. Compute embeddings with fastembed
                embeddings = embedding_model.embed(chunks)

                # 4. Index documents in BeaverDB
                for text, embedding in zip(chunks, embeddings):
                    doc = Document(
                        embedding=embedding.tolist(),
                        text=text,
                        source_file=file_name
                    )
                    docs_collection.index(doc)
                    total_chunks_indexed += 1

            except Exception as e:
                st.error(f"Failed to process {file_name}: {e}")

        overall_progress.progress(1.0, text="Indexing complete!")
        st.success(f"Success! Added **{total_chunks_indexed}** new text chunks to the knowledge base.")
    else:
        st.warning("Please upload at least one file to index.")
