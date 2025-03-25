# This program is intended to create a Chatbot that accesses a FAISS Vector database that contains a large HR website 
# with tons of HR policies, practices and domain knowledge. The ChatBot will give the user the ability to query on any 
# HR related information in a conversation form with conversational meory like Chat GPT. 
# The UI of the Chat Bot is done using the Streamlit Library.
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_core.language_models.llms import LLM

load_dotenv()

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()

class CustomChatModel(LLM):
    def _call(self, prompt: str, stop: List[str] = None, **kwargs) -> str:
        """Process the prompt and return a response."""
        if "Context:" in prompt:
            context = prompt.split("Context:")[1].split("Question:")[0].strip()
            question = prompt.split("Question:")[1].strip()
            return f"Based on the HR policies:\n\n{context}\n\nRegarding your question about '{question}'"
        return f"Based on the HR policies, here's what I found about: {prompt}"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"name": "CustomChatModel"}
    
    @property
    def _llm_type(self) -> str:
        return "custom"

def build_chat_history(chat_history_list):
    chat_history = []
    for message in chat_history_list:        
        chat_history.append(HumanMessage(content=message[0]))
        chat_history.append(AIMessage(content=message[1]))
    return chat_history

def query(question, chat_history):
    try:
        chat_history = build_chat_history(chat_history)
        embeddings = CustomEmbeddings()
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = CustomChatModel()

        # Get relevant documents first
        retriever = new_db.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(question)
        context = "\n".join(doc.page_content for doc in docs)

        # Create the response using the context
        response = llm(f"Context: {context}\nQuestion: {question}")
        
        return {"answer": response}
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return {"answer": "I apologize, but I encountered an error while processing your question. Please try again."}

def show_ui():
    st.title("🤖 HR Policy Assistant")    
    st.subheader("Ask about HR policies and procedures")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your HR-related question: "):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Searching HR policies..."):
            response = query(prompt, st.session_state.chat_history)
            
            with st.chat_message("assistant"):
                st.markdown(response["answer"])

            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.session_state.chat_history.append((prompt, response["answer"]))

if __name__ == "__main__":
    show_ui()