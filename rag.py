import os
import streamlit as st
import tempfile
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCK7QRKCq4GuBZErZaMLRInRjmT891g6Bg"

# Initialize the Gemini chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Streamlit UI
st.title("📄 LangChain RAG Chatbot with Gemini")
st.markdown("Upload a PDF and ask questions based on its content.")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "history" not in st.session_state:
    st.session_state.history = []

# PDF Upload and Processing
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()

    # Chunk the PDF content
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)

    # Create LangChain documents
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Create FAISS vectorstore from PDF
    st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
    st.success("✅ PDF processed and indexed.")

# Question input
user_question = st.text_input("Ask a question about the PDF:")

if st.button("Ask") and user_question:
    if not st.session_state.vectorstore:
        st.warning("Please upload a PDF first.")
    else:
        # Perform retrieval
        retriever = st.session_state.vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(user_question)

        # Load chain for QA
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        result = chain.run(input_documents=relevant_docs, question=user_question)

        # Store in history
        st.session_state.history.append({"user": user_question, "bot": result})

# Display chat history
if st.session_state.history:
    st.markdown("### 💬 Chat History")
    for chat in reversed(st.session_state.history):
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")
