# --------------------------------------
# ✅ STEP 1: Import Required Libraries
# --------------------------------------
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
# --------------------------------------
# ✅ STEP 2: Streamlit UI
# --------------------------------------
st.header("📄 PDF Chatbot - Ask Questions from Your PDF")

with st.sidebar:
    st.title("📚 Upload PDF")
    file = st.file_uploader("Upload a PDF file", type="pdf")

# --------------------------------------
# ✅ STEP 3: Extract and Chunk PDF
# --------------------------------------
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # --------------------------------------
    # ✅ STEP 4: Generate Free Embeddings using Hugging Face
    # --------------------------------------
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    # --------------------------------------
    # ✅ STEP 5: User Input and Vector Search
    # --------------------------------------
    user_query = st.text_input("💬 Ask something about the PDF")

    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        # --------------------------------------
        # ✅ STEP 6: Groq LLM Setup
        # --------------------------------------
        GROQ_API_KEY = os.getenv["GROQ_API_KEY"]  # Replace with your Groq API Key
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="mixtral-8x7b-32768",  # Or "llama3-70b-8192", "gemma-7b-it"
            temperature=0,
            max_tokens=300
        )

        # --------------------------------------
        # ✅ STEP 7: Define Prompt Template
        # --------------------------------------
        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful AI assistant. Use the provided context from the PDF to answer the question accurately.
            If the answer cannot be found, say: "I'm sorry, I couldn't find that information in the uploaded PDF."

            ---------------------
            Context:
            {context}

            Question:
            {input}
            ---------------------
            Answer:
            """
        )

        chain = create_stuff_documents_chain(llm, prompt)

        # --------------------------------------
        # ✅ STEP 8: Generate and Display Answer
        # --------------------------------------
        output = chain.invoke({
            "input": user_query,
            "input_documents": matching_chunks
        })

        st.subheader("🧠 Answer")
        st.write(output)
