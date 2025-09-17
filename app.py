import streamlit as st

st.set_page_config(
    page_title="RAG Chatbot with BeaverDB",
    page_icon="🤖"
)

st.title("🤖 RAG Chatbot with BeaverDB")

st.markdown("""
Welcome to your document-aware chatbot!

This application uses the power of Retrieval-Augmented Generation (RAG) to answer questions based on your own documents.

**👈 Select a page from the sidebar to get started:**

- **Index Documents**: Upload your `.docx` or `.pdf` files to build the knowledge base.
- **Search Documents**: Directly query the indexed knowledge base to test retrieval.
- **RAG Chatbot**: Chat with an AI assistant that uses the indexed documents to answer questions.
""")
