import streamlit as st
import cohere
import uuid
import hnswlib
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
import base64
import io

# Initialize Cohere client
cohere_key = st.secrets["api"]
co = cohere.Client(cohere_key)

# Raw documents
raw_documents = [
    {"title": "FrenpetBranding", "url": "https://docs.frenpet.xyz/branding"},
    {"title": "Frenpetcontracts", "url": "https://docs.frenpet.xyz/contracts"},
    {"title": "FrenpetPgold", "url": "https://docs.frenpet.xyz/pgold"},
    {"title": "FrenpetFP", "url": "https://docs.frenpet.xyz/fp"},
    {"title": "FrenpetRewards", "url": "https://docs.frenpet.xyz/rewards"},
    {"title": "FrenpetMP", "url": "https://docs.frenpet.xyz/marketplace"},
    {"title": "FrenpetQuests", "url": "https://docs.frenpet.xyz/quests"},
    {"title": "FrenpetPVP", "url": "https://docs.frenpet.xyz/pvp"},
    {"title": "FrenpetStake", "url": "https://docs.frenpet.xyz/stake"},
    {"title": "FrenpetFree", "url": "https://docs.frenpet.xyz/freemium"},
    {"title": "FrenpetGameplay", "url": "https://docs.frenpet.xyz/gameplay"},
    {"title": "FrenpetDoc", "url": "https://docs.frenpet.xyz/"}
]

# Vectorstore class (unchanged)
class Vectorstore:
    def __init__(self, raw_documents: List[Dict[str, str]]):
        self.raw_documents = raw_documents
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.load_and_chunk()
        self.embed()
        self.index()

    def load_and_chunk(self) -> None:
        print("Loading documents...")
        for raw_document in self.raw_documents:
            elements = partition_html(url=raw_document["url"])
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                self.docs.append(
                    {
                        "title": raw_document["title"],
                        "text": str(chunk),
                        "url": raw_document["url"],
                    }
                )

    def embed(self) -> None:
        print("Embedding document chunks...")
        batch_size = 90
        self.docs_len = len(self.docs)
        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i : min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        print("Indexing document chunks...")
        self.idx = hnswlib.Index(space="ip", dim=1024)
        self.idx.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))
        print(f"Indexing complete with {self.idx.get_current_count()} document chunks.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        query_emb = co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        ).embeddings
        
        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]
        rank_fields = ["title", "text"]
        docs_to_rerank = [self.docs[doc_id] for doc_id in doc_ids]
        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v3.0",
            rank_fields=rank_fields
        )

        doc_ids_reranked = [doc_ids[result.index] for result in rerank_results.results]
        docs_retrieved = []
        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )

        return docs_retrieved

# Function to run the chatbot
def run_chatbot(message, chat_history=None):
    if chat_history is None:
        chat_history = []
    
    # Format chat history for Cohere API
    cohere_chat_history = [
        {"role": str(entry["role"]), "message": str(entry["content"])}
        for entry in chat_history
        if isinstance(entry, dict) and "role" in entry and "content" in entry
    ]
    
    # Generate search queries
    response = co.chat(
        message=message,
        model="command-r-plus",
        search_queries_only=True,
        chat_history=cohere_chat_history  # Use properly formatted history
    )
    
    search_queries = []
    for query in response.search_queries:
        search_queries.append(query.text)

    # If there are search queries, retrieve the documents
    if search_queries:
        print("Retrieving information...", end="")  # Optional print for debugging
        documents = []
        for query in search_queries:
            documents.extend(vectorstore.retrieve(query))

        # Use document chunks to respond
        response = co.chat_stream(
            message=message,
            model="command-r-plus",
            documents=documents,
            chat_history=cohere_chat_history,
        )
    else:
        response = co.chat_stream(
            message=message,
            model="command-r-plus",
            chat_history=cohere_chat_history,
        )
        
    # Print the chatbot response, citations, and documents
    chatbot_response = ""
    print("\nChatbot:")  # Optional print for debugging
    for event in response:
        if event.event_type == "text-generation":
            print(event.text, end="")
            chatbot_response += event.text
        if event.event_type == "stream-end":
            if event.response.citations:
                print("\n\nCITATIONS:")
                for citation in event.response.citations:
                    print(citation)
            if event.response.documents:
                print("\nCITED DOCUMENTS:")
                for document in event.response.documents:
                    print(document)
            # Update the chat history for the next turn
            chat_history = event.response.chat_history

    return chatbot_response, chat_history  # Return both the response and updated chat history

# Initialize session state for documents and vectorstore
if 'vectorstore' not in st.session_state:
    print("Loading documents...")
    st.session_state.vectorstore = Vectorstore(raw_documents)

# Use the stored vectorstore instead of creating a new one
vectorstore = st.session_state.vectorstore


# Define avatar images with different sizes
USER_AVATAR = r"fpuser.png"  # Replace with your image path or URL
BOT_AVATAR = r"fpdragon.png"  # Replace with your image path or URL

# Add this near the top of your script
st.markdown("""
    <style>
    [data-testid="stChatMessage"] {
        padding: 3rem;
    }
    
    /* Increase size of assistant's avatar */
    [data-testid="stChatMessage"] [data-testid="stImage"].assistant {
        width: 100px !important;
        height: 100px !important;
    }
    
    /* Keep user's avatar smaller */
    [data-testid="stChatMessage"] [data-testid="stImage"].user {
        width: 30px !important;
        height: 30px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app layout (Updated to resemble ChatGPT UI)
st.title("FP Dragon the Onboarder")

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, dict) and "role" in message and "content" in message:
        avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(str(message["role"]), avatar=avatar):
            st.markdown(f'<div class="{message["role"]}">{str(message["content"])}</div>', 
                       unsafe_allow_html=True)

# Handle user input
if prompt := st.chat_input("Ask something about Frenpet:"):
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        chatbot_response, updated_history = run_chatbot(
            prompt, 
            st.session_state.chat_history
        )
        st.markdown(chatbot_response)
        
        # Update chat history
        st.session_state.chat_history = updated_history

