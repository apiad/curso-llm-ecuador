import streamlit as st
from beaver import BeaverDB
from fastembed import TextEmbedding
import os

# --- Configuration ---
DB_PATH = "knowledge_base.db"
COLLECTION_NAME = "documents"

# --- Helper Functions (cached for performance) ---

@st.cache_resource
def get_db():
    """Initializes and returns the BeaverDB instance."""
    if not os.path.exists(DB_PATH):
        st.error(f"Database file not found at {DB_PATH}. Please index some documents first on the 'Index Documents' page.")
        return None
    return BeaverDB(DB_PATH)

@st.cache_resource
def get_embedding_model():
    """Loads and caches the fastembed model."""
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- Streamlit UI ---

st.set_page_config(page_title="Search Documents", page_icon="🔎")
st.title("🔎 Search Indexed Documents")
st.markdown("Directly query the vector database to see which document chunks are most relevant to your search term.")

db = get_db()
embedding_model = get_embedding_model()

if db:
    query = st.text_input("Enter your search query:", placeholder="e.g., What is the main idea of the document?")
    top_k = st.slider("Number of results to retrieve (k):", min_value=1, max_value=20, value=5)

    if query:
        with st.spinner("Searching for relevant documents..."):
            try:
                # 1. Embed the query
                query_embedding = list(embedding_model.embed(query))[0]

                # 2. Search the BeaverDB collection
                docs_collection = db.collection(COLLECTION_NAME)
                search_results = docs_collection.search(vector=query_embedding.tolist(), top_k=top_k)

                st.success(f"Found {len(search_results)} results.")

                # 3. Display the results
                for i, (doc, distance) in enumerate(search_results):
                    with st.expander(f"**Result {i+1}** | Source: `{doc.source_file}` | Distance: `{distance:.4f}`"):
                        st.write(
                            doc.text,
                        )

            except Exception as e:
                st.error(f"An error occurred during search: {e}")
