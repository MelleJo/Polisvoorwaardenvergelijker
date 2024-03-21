import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from pydantic import BaseModel, Field

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

st.title("Document Comparison Tool")
doc1 = st.file_uploader("Upload the first document (PDF):", type="pdf", key="doc1")
doc2 = st.file_uploader("Upload the second document (PDF):", type="pdf", key="doc2")
question = st.text_input("Enter your question for comparison:")

# Adapted function to extract text from an uploaded PDF in memory
def extract_text_from_pdf_by_page(uploaded_file):
    pages_text = []
    if uploaded_file is not None:
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return pages_text

# Updated function to process an uploaded document using the adapted text extraction logic
def process_uploaded_document(uploaded_file):
    pages_text = extract_text_from_pdf_by_page(uploaded_file)
    if pages_text:
        document_text = " ".join(pages_text)
        embeddings = OpenAIEmbeddings()
        document = Document(page_content=document_text)
        retriever = FAISS.from_documents([document], embeddings).as_retriever()
        return retriever
    return None


def run_comparison_agent(question, tools):
    if question and tools:
        class DocumentInput(BaseModel):
            question: str

        agent = initialize_agent(
            agent=AgentType.OPENAI_FUNCTIONS,
            tools=tools,
            llm=llm,
            verbose=True,
        )
        response = agent({"input": question})
        return response
    return "Please provide a question and upload both documents."

if st.button("Compare Documents"):
    tools = []
    documents = [(doc1, "doc1"), (doc2, "doc2")]

    for uploaded_file, name in documents:
        retriever = process_uploaded_document(uploaded_file)
        if retriever:
            tools.append(
                Tool(
                    args_schema=DocumentInput,
                    name=name,
                    description=f"Useful when you want to answer questions about {name}",
                    func=lambda question: RetrievalQA.from_chain_type(llm=llm, retriever=retriever)(question),
                )
            )

    if len(tools) == 2 and question:
        result = run_comparison_agent(question, tools)
        st.write("Comparison Result:", result)
    else:
        st.error("Please upload both documents and enter a question.")
