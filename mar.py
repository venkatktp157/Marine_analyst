import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables (.env must contain GROQ_API_KEY)
load_dotenv()

# ---------------------------
# Initialize Groq LLM
# ---------------------------
llm = ChatGroq(model="llama3-70b-8192", temperature=0)

# Prompt template for Marine Data Analyst role
prompt = PromptTemplate.from_template(
    "You are a Marine Data Analyst. Analyze the following content "
    "and provide insights relevant to marine science and operations:\n\n{content}\n\n"
)

# Modern pipeline style (no LLMChain needed)
chain = prompt | llm

# Wrap into a Tool for agent use
summarize_tool = Tool.from_function(
    func=lambda content: chain.invoke({"content": content}),
    name="MarineDataSummarizer",
    description="Analyzes and summarizes marine-related documents"
)

# Initialize agent
agent = initialize_agent(
    tools=[summarize_tool],
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True
)

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
    type=["pdf", "docx", "doc", "xlsx", "xls", "png", "jpeg", "jpg", "bmp"]
)
user_query = st.text_area("Enter your query (e.g., 'Summarize salinity trends')")

if uploaded_file and user_query:
    # Save uploaded file temporarily
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    chunks = load_and_process_document(uploaded_file.name)
    content = " ".join([chunk.page_content for chunk in chunks[:5]])  # first 5 chunks

    prompt_text = f"{user_query}\n\nDocument content:\n{content}"
    response = agent.invoke(prompt_text)

    st.subheader("Marine Data Analyst Response")
    st.write(response)
