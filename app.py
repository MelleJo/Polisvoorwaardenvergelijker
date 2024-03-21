import streamlit as st
from io import BytesIO
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from PyPDF2 import PdfReader  # Assuming use of PyPDF2 for PDF processing
from pydantic import BaseModel, Field

class DocumentInput(BaseModel):
    question: str = Field()

# Initialize the large language model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Streamlit UI for uploading documents
st.title("Document Comparison Tool")

doc1 = st.file_uploader("Upload first document (PDF):", type=['pdf'], key="doc1")
doc2 = st.file_uploader("Upload second document (PDF):", type=['pdf'], key="doc2")
question = st.text_input("Enter your question:")

# Function to process uploaded documents
def process_document(uploaded_file):
    if uploaded_file is not None:
        reader = PdfReader(BytesIO(uploaded_file.read()))
        pages = [page.extract_text() for page in reader.pages]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        embeddings = OpenAIEmbeddings()
        retriever = FAISS.from_documents(docs, embeddings).as_retriever()
        return retriever
    return None

if st.button("Compare Documents"):
    if doc1 and doc2 and question:
        tools = []
        documents = [(doc1, "doc1"), (doc2, "doc2")]

        for uploaded_file, name in documents:
            retriever = process_document(uploaded_file)
            if retriever:
                tools.append(
                    Tool(
                        args_schema=DocumentInput,
                        name=name,
                        description=f"useful when you want to answer questions about {name}",
                        func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
                    )
                )

        if len(tools) == 2:
            # Initialize comparison agent with the tools
            from langchain.agents import initialize_agent, AgentType

            agent = initialize_agent(
                agent=AgentType.OPENAI_FUNCTIONS,
                tools=tools,
                llm=llm,
                verbose=True,
            )

            result = agent({"input": question})
            st.write("Comparison Result:", result)
        else:
            st.error("Error in processing documents.")
    else:
        st.error("Please upload both documents and enter a question.")
