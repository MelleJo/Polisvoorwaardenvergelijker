from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field
from langchain.agents import AgentType, initialize_agent

class DocumentInput(BaseModel):
    question: str = Field()


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

tools = []
files = [
    doc1 = st.FileUploader(label=file["name"], key=file["name"])
    doc2 = st.FileUploader(label=file["name"], key=file["name"])

]

for file in files:
    loader = PyPDFLoader(file[doc1, doc2])
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=file["name"],
            description=f"useful when you want to answer questions about {file['name']}",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
        )
    )

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-0613",
)

agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=tools,
    llm=llm,
    verbose=True,
)

agent({"input": "did alphabet or tesla have more revenue?"})