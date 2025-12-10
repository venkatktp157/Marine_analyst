import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader,
    CSVLoader   # <-- add this
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------
# Initialize Groq LLM using Streamlit secrets
# ---------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=st.secrets["GROQ_API_KEY"]
)

# Prompt template
analysis_prompt = PromptTemplate.from_template(
    "You are a Marine Data Analyst. Analyze the following content "
    "and provide insights relevant to marine science and operations:\n\n{content}\n\n"
)

summary_prompt = PromptTemplate.from_template(
    "Summarize the following marine data document into a concise report "
    "highlighting key findings, trends, and recommendations:\n\n{content}\n\n"
)

analysis_chain = analysis_prompt | llm
summary_chain = summary_prompt | llm

# ---------------------------
# Helper: Load and process document
# ---------------------------
def load_and_process_document(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx") or file_path.endswith(".doc"):
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        loader = UnstructuredExcelLoader(file_path)
    elif file_path.endswith((".png", ".jpeg", ".jpg", ".bmp")):
        loader = UnstructuredImageLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)   # <-- new branch    
    else:
        raise ValueError("Unsupported file type")

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    return chunks

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌊 Marine Data Analyst Agent")

uploaded_file = st.file_uploader(
    "Upload a marine data file (PDF, Word, Excel, Image)",
    type=["pdf", "docx", "doc", "xlsx", "xls", "csv","png", "jpeg", "jpg", "bmp"]
)
user_query = st.text_area("Enter your query (e.g., 'Summarize speed trends')")

if uploaded_file and user_query:
    # Save uploaded file temporarily
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    chunks = load_and_process_document(uploaded_file.name)
    content = " ".join([chunk.page_content for chunk in chunks[:5]])  # first 5 chunks

    # Run analysis
    analysis_response = analysis_chain.invoke({"content": f"{user_query}\n\nDocument content:\n{content}"})
    st.subheader("Marine Data Analyst Response")
    st.write(analysis_response)

    # Run summarization
    summary_response = summary_chain.invoke({"content": content})

    # Ensure it's a string
    if not isinstance(summary_response, str):
        summary_text = str(summary_response)
    else:
        summary_text = summary_response

    st.subheader("📄 Document Summary")
    st.write(summary_text)
    
    # Download button for summary
    st.download_button(
        label="⬇️ Download Summary",
        data=summary_text.encode("utf-8"),   # convert to bytes
        file_name="marine_summary.txt",
        mime="text/plain"
    )