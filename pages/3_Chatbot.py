import streamlit as st
from datetime import datetime
import pydantic
from beaver import BeaverDB
from fastembed import TextEmbedding
import argo
from argo import Message
from argo.client import stream
from argo.skills import chat
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


# --- Configuration ---
DB_PATH = "knowledge_base.db"
COLLECTION_NAME = "documents"

# --- Helper Functions (cached for performance) ---


@st.cache_resource
def get_db():
    """Initializes and returns the BeaverDB instance."""
    if not os.path.exists(DB_PATH):
        return None
    return BeaverDB(DB_PATH)


@st.cache_resource
def get_embedding_model():
    """Loads and caches the fastembed model."""
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def search_knowledge_base(query: str) -> List[str]:
    """
    Searches the BeaverDB collection for the top 3 most relevant document chunks.
    """
    db = get_db()
    embedding_model = get_embedding_model()
    if not db or not embedding_model:
        return []

    query_embedding = list(embedding_model.embed(query))[0]
    docs_collection = db.collection(COLLECTION_NAME)

    search_results = docs_collection.search(vector=query_embedding.tolist(), top_k=5)

    return [doc for doc, distance in search_results]


class Summary(pydantic.BaseModel):
    summary: str
    relevant: bool

class Event(pydantic.BaseModel):
    title: str
    date: datetime
    description: str


class Calendar:
    def __init__(self, *events):
        self.events = list(events)


@st.cache_resource
def initialize_agent():
    """
    Initializes and configures the ARGO ChatAgent with RAG skills.
    """
    # Make sure you have your LLM provider's API key in an .env file
    if not os.getenv("TOKEN"):
        st.error("TOKEN environment variable not set. Please create a .env file.")
        return None

    llm = argo.LLM(
        model="google/gemini-2.5-flash",
        api_key=os.getenv("TOKEN"),
        base_url="https://openrouter.ai/api/v1",
        verbose=True,
    )

    agent = argo.ChatAgent(
        name="Manolo",
        description="A helpful assistant that answers questions based on a document knowledge base.",
        llm=llm,
        skills=[chat],  # Pre-built casual chat skill
    )

    agent.calendar = Calendar(
        Event(title="Meeting with Bob", date=datetime(2025, 9, 18, 15, 0), description="Discuss project updates."),
        Event(title="Dentist Appointment", date=datetime(2025, 10, 10, 10, 30), description="Routine check-up."),
    )

    @agent.skill
    async def handle_calendar(ctx: argo.Context):
        """
        Handles calendar-related queries.
        Use this for general questions about dates, days of the week, months, etc.
        The schedule of user events, add new events, and list existing events.
        """

        ctx.add(Message.system(f"Today is: {datetime.now().strftime('%Y-%m-%d')}"))

        tool = await ctx.equip(tools=[list_calendar, add_event])
        result = await ctx.invoke(tool)

        ctx.add(Message.system(result))

        await ctx.reply()


    @agent.tool
    async def add_event(title: str, date: datetime, description: str):
        """
        Adds a new event to the user calendar.
        """

        agent.calendar.events.append(
            Event(title=title, date=date, description=description)
        )

        return f"Event '{title}' added on {date.strftime('%Y-%m-%d %H:%M')}."


    @agent.tool
    async def list_calendar(from_date: datetime):
        """Lists all events in the user's calendar."""
        events = agent.calendar.events

        if not events:
            return "No events scheduled."

        event_list = "\n".join(
            f"- {event.title} on {event.date.strftime('%Y-%m-%d %H:%M')}: {event.description}"
            for event in events if event.date.timetuple() >= from_date.timetuple()
        )

        return event_list

    @agent.skill
    async def question_answering(ctx: argo.Context):
        """
        Answers user questions that require knowledge from the indexed documents.
        Use this for any specific questions that cannot be answered with general knowledge.
        """
        result = await ctx.invoke(search_kb)

        # 2. Add the context to the conversation for the LLM
        system_prompt = f"""
        You are a helpful assistant. Please answer the user's question based ONLY on the
        following context. If the context does not contain the answer, state that you
        could not find the information in the provided documents.

        Beside every claim, put the source filename in parentheses.

        Context:
        {result.result}
        """

        ctx.add(argo.Message.system(system_prompt))

        # 3. Generate the reply
        await ctx.reply("Reply with the information in the context.")

    @agent.tool
    async def search_kb(query: str):
        """
        Finds relevant documents in the knowledge base.
        """
        # 1. Retrieve context from BeaverDB
        retrieved_docs = search_knowledge_base(query)

        if not retrieved_docs:
            return "I couldn't find any relevant information in the indexed documents to answer your question."

        context = []

        for doc in retrieved_docs:
            summary = await llm.create(
                model=Summary,
                messages=[
                    Message.system(
                        f"Summarize the following text in a concise manner given the user query, and determine if its relevant for the user query.\n\nQuery: {query}."
                    ),
                    Message.user(doc.text),
                ],
            )

            if summary.relevant:
                context.append(dict(text=summary.summary, source_file=doc.source_file))

        return "\n\n---\n\n".join(
            "Filename: {}\n\nContent:\n{}".format(doc["source_file"], doc["text"])
            for doc in context
        )

    return agent


# --- Streamlit UI ---

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")
st.markdown("Chat with an AI assistant that uses your indexed documents for answers.")

# Initialize agent and database
agent = initialize_agent()
db = get_db()

if not db:
    st.warning(
        "Database not found. Please go to the 'Index Documents' page and upload some files first."
    )
elif not agent:
    st.error("Agent could not be initialized. Please check your environment variables.")
else:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use argo.client.stream and st.write_stream for a clean implementation
                response_generator = stream(agent, prompt)
                full_response = st.write_stream(response_generator)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
