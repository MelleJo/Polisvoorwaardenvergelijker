# Import necessary libraries
import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.agents import Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field
from langchain.agents import AgentType

# Define a Pydantic model for the input
class DocumentInput(BaseModel):
    question: str = Field()

# Initialize the large language model with GPT-4 and API key from st.secrets
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4-turbo-preview",
    api_key=st.secrets["OPENAI_API_KEY"]
)

def load_and_prepare_document(uploaded_file):
    """Loads, splits, and prepares a document for retrieval using PyPDF2 from an uploaded file."""
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pages = [page.extract_text() for page in reader.pages if page.extract_text() is not None]

    # Assuming CharacterTextSplitter expects a list of strings (direct text) without needing a 'page_content' attribute
    # Directly use 'pages' as it now holds a list of text extracted from each PDF page.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)  # Directly pass the list of text chunks

    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()
    return retriever


def compare_documents(question, tools):
    """Compares documents based on a question and returns the result."""
    agent = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=True,
    )
    return agent({"input": question})

# Streamlit UI for uploading documents
st.title("Insurance Policy Document Comparison")

doc1 = st.file_uploader("Upload first insurance policy document (PDF):", type=['pdf'])
doc2 = st.file_uploader("Upload second insurance policy document (PDF):", type=['pdf'])
question = st.text_input("Enter your question:")

if st.button("Compare Documents"):
    if doc1 and doc2 and question:
        tools = []

        # Process the first document
        retriever1 = load_and_prepare_document(doc1)
        tools.append(
            Tool(
                args_schema=DocumentInput,
                name="doc1",
                description="Insurance policy document 1",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever1),
            )
        )

        # Process the second document
        retriever2 = load_and_prepare_document(doc2)
        tools.append(
            Tool(
                args_schema=DocumentInput,
                name="doc2",
                description="Insurance policy document 2",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever2),
            )
        )

        # Compare the documents
        result = compare_documents(question, tools)
        st.write("Comparison Result:", result)
    else:
        st.write("Please upload both documents and enter a question.")
