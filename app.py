import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field

# Define input model for questions
class DocumentInput(BaseModel):
    question: str = Field()

# Initialize the LLM with your specific model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Streamlit UI for uploading PDF documents
st.title("Document Comparison Tool")
doc1 = st.file_uploader("Upload the first document (PDF):", type="pdf", key="doc1")
doc2 = st.file_uploader("Upload the second document (PDF):", type="pdf", key="doc2")
question = st.text_input("Enter your question for comparison:")

# Function to process an uploaded document
def process_uploaded_document(uploaded_file):
    if uploaded_file is not None:
        # Use BytesIO to read the uploaded file in memory
        reader = PdfReader(BytesIO(uploaded_file.read()))
        pages = [page.extract_text() for page in reader.pages if page.extract_text() is not None]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        embeddings = OpenAIEmbeddings()
        retriever = FAISS.from_documents(docs, embeddings).as_retriever()
        return retriever
    return None

# Function to initialize and run the comparison agent
def run_comparison_agent(question, tools):
    if question and tools:
        agent = initialize_agent(
            agent=AgentType.OPENAI_FUNCTIONS,
            tools=tools,
            llm=llm,
            verbose=True,
        )
        response = agent({"input": question})
        return response
    return "Please provide a question and upload both documents."

# Main comparison logic when the 'Compare Documents' button is clicked
if st.button("Compare Documents"):
    tools = []
    documents = [(doc1, "doc1"), (doc2, "doc2")]

    for uploaded_file, name in documents:
        retriever = process_uploaded_document(uploaded_file)
        if retriever:
            # Wrap retrievers in a Tool for each document
            tools.append(
                Tool(
                    args_schema=DocumentInput,
                    name=name,
                    description=f"Useful when you want to answer questions about {name}",
                    func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
                )
            )

    # Run the comparison agent if both documents are uploaded and a question is provided
    if len(tools) == 2 and question:
        result = run_comparison_agent(question, tools)
        st.write("Comparison Result:", result)
    else:
        st.error("Please upload both documents and enter a question.")
