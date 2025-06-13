import streamlit as st
import PyPDF2
import google.generativeai as genai

# Initialize Gemini
genai.configure(api_key="AIzaSyCK7QRKCq4GuBZErZaMLRInRjmT891g6Bg")

# Set up Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

# PDF text extractor
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = ""

# Streamlit UI
st.title("📄 RAG Chatbot with Gemini")
st.markdown("Upload a PDF and ask questions based on its content.")

# PDF upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded_file is not None:
    st.session_state.pdf_content = extract_text_from_pdf(uploaded_file)
    st.success("PDF content extracted and ready!")

# Chat input
user_input = st.text_input("Ask a question about the PDF:")

# Generate response
if st.button("Ask") and user_input:
    if not st.session_state.pdf_content:
        st.error("Please upload a PDF file first.")
    else:
        prompt = f"""
You are an AI assistant. Answer the user's question based **only** on the information provided in the document below.

Document:
\"\"\"
{st.session_state.pdf_content}
\"\"\"

Question:
{user_input}
        """

        response = model.generate_content(prompt)
        answer = response.text

        # Store conversation
        st.session_state.history.append({"user": user_input, "bot": answer})

# Display chat history
if st.session_state.history:
    st.markdown("### 💬 Chat History")
    for i, chat in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")
