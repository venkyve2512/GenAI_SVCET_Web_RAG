import streamlit as st
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.documents import Document
import time

# Load environment variables
load_dotenv()

# Set up API keys from environment variables

#groq_api_key = os.getenv("GROQ_API_KEY")
#os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")


st.set_page_config(page_title="Web RAG with Groq & SerpApi", layout="wide")
st.image('PragyanAI_Transperent_github.png')
st.title("Web RAG: Chat with Websites and SerpApi Search")

# Initialize session state for vector store and chat history
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for user input
with st.sidebar:
    st.header("Input Source")
    website_url = st.text_input("Enter Website URL to scrape")
    search_query = st.text_input("Enter SerpApi Search Query")
    
    if st.button("Process"):
        if not website_url and not search_query:
            st.warning("Please enter a website URL or a search query.")
        else:
            with st.spinner("Processing..."):
                docs = []
                # Process website URL if provided
                if website_url:
                    try:
                        loader = WebBaseLoader(website_url)
                        web_docs = loader.load()
                        docs.extend(web_docs)
                        st.success(f"Successfully scraped {website_url}")
                    except Exception as e:
                        st.error(f"Failed to scrape website: {e}")

                # Process SerpApi search query if provided
                if search_query:
                    try:
                        search = SerpAPIWrapper()
                        search_results = search.results(search_query)
                        
                        # Check for organic results and process them
                        if "organic_results" in search_results:
                            search_docs = [
                                Document(
                                    page_content=result.get("snippet", ""),
                                    metadata={"source": result.get("link", ""), "title": result.get("title", "")}
                                ) for result in search_results["organic_results"]
                            ]
                            docs.extend(search_docs)
                            st.success(f"Successfully fetched search results for '{search_query}'")
                        else:
                            st.warning("No organic results found for the search query.")

                    except Exception as e:
                        st.error(f"Failed to perform SerpApi search: {e}")

                # Process the combined documents
                if docs:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    final_documents = text_splitter.split_documents(docs)

                    # Use a pre-trained model from Hugging Face for embeddings
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    st.session_state.vector = FAISS.from_documents(final_documents, embeddings)
                    st.success("Vector store created successfully!")
                else:
                    st.warning("No content was processed.")


# Main chat interface
st.header("Chat with the Web")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Create the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context from the website or search results.
    Please provide the most accurate and comprehensive response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Display previous chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt_input := st.chat_input("Ask a question..."):
    if st.session_state.vector is not None:
        with st.chat_message("user"):
            st.markdown(prompt_input)
        
        st.session_state.chat_history.append({"role": "user", "content": prompt_input})

        with st.spinner("Thinking..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({"input": prompt_input})
            response_time = time.process_time() - start

            with st.chat_message("assistant"):
                st.markdown(response['answer'])
                st.info(f"Response time: {response_time:.2f} seconds")

            st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

    else:
        st.warning("Please process a URL or search query first.")
