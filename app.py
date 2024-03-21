import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

# Assuming your API key is stored in Streamlit's secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

class DocumentInput(BaseModel):
    question: str = Field()

def load_and_prepare_document(uploaded_file):
    """Extracts text from uploaded PDF and prepares it for analysis."""
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    text_chunks = [page.extract_text() or "" for page in reader.pages]
    return text_chunks

def initialize_comparison_agent():
    """Initializes the comparison agent with OpenAI and document processing tools."""
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview")
    return llm

def compare_documents(doc1_text, doc2_text, question, llm):
    """Compares two documents based on a question using the LangChain RetrievalQA chain."""
    # Combine text from both documents for embedding and indexing
    combined_text = doc1_text + doc2_text
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    retriever = FAISS.from_texts(combined_text, embeddings).as_retriever()
    
    # Prepare the RetrievalQA tool with the retriever and llm
    tool = RetrievalQA(llm=llm, retriever=retriever)
    
    # Use the tool to answer the question based on the combined documents
    answer = tool({'question': question})
    return answer

# Streamlit UI for uploading documents and asking questions
st.title("Insurance Policy Document Comparison")

doc1 = st.file_uploader("Upload the first insurance policy document (PDF):", type=['pdf'])
doc2 = st.file_uploader("Upload the second insurance policy document (PDF):", type=['pdf'])
question = st.text_input("Enter your question for comparison:")

if st.button("Compare Documents"):
    if doc1 and doc2 and question:
        # Load and prepare both documents
        doc1_text = load_and_prepare_document(doc1)
        doc2_text = load_and_prepare_document(doc2)
        
        # Initialize LangChain's comparison agent
        llm = initialize_comparison_agent()
        
        # Compare the documents and display the answer
        answer = compare_documents(doc1_text, doc2_text, question, llm)
        st.write("Comparison Result:", answer)
    else:
        st.error("Please upload both documents and enter a question.")
