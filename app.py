import os
import streamlit as st
from PyPDF2 import PdfReader
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

# Initialize the large language model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

def load_and_prepare_document(file_path):
    """Loads, splits, and prepares a document for retrieval using PyPDF2."""
    reader = PdfReader(file_path)
    pages = [page.extract_text() for page in reader.pages if page.extract_text() is not None]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
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

# Streamlit UI
st.title("Insurance Policy Document Comparison")

doc1_path = st.text_input("Enter path to first insurance policy document (PDF):")
doc2_path = st.text_input("Enter path to second insurance policy document (PDF):")
question = st.text_input("Enter your question:")

if st.button("Compare Documents"):
    if doc1_path and doc2_path and question:
        tools = []

        # Process the first document
        retriever1 = load_and_prepare_document(doc1_path)
        tools.append(
            Tool(
                args_schema=DocumentInput,
                name="doc1",
                description="Insurance policy document 1",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever1),
            )
        )

        # Process the second document
        retriever2 = load_and_prepare_document(doc2_path)
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
        st.write("Please provide paths to both documents and a question.")

