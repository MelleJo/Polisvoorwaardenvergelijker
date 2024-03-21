import streamlit as st
import os

from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field
from langchain.agents import AgentType, initialize_agent
from PyPDF2 import PdfReader


def extract_text_from_pdf_by_page(file_path):
    pages_text = []
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return pages_text

def process_document(document_path, user_question):
    with st.spinner('Denken...'):
        # Extract text from the document
        document_pages = extract_text_from_pdf_by_page(document_path)
        if not document_pages or all(page.strip() == "" for page in document_pages):
        
            st.error("No valid text extracted from the document. Please check the document format or content.")
            return

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(document_pages, embeddings)
        docs = knowledge_base.similarity_search(user_question)
        document_text = " ".join([doc.page_content for doc in docs])

        template = """
        Analyseer de vraag van de gebruiker: {user_question} en vergelijk vervolgens de volgende twee polisvoorwaarden {document_text_doc1} en {document_text_doc2} met elkaar en beantwoord de vraag van de gebruiker. 
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        
        # Perform similarity search
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview", temperature=0, streaming=True)
        chain = prompt | llm | StrOutputParser() 
        return chain.stream({
            "document_text_doc1": document_text,
            "document_text_doc2": document_text,
            "user_question": user_question,
        })

def main():
    st.title("Polisvoorwaardenvergelijker - testversie 0.1.")
    doc1 = st.file_uploader("Upload Insurance Policy Document 1", type=['txt', 'pdf'])
    doc2 = st.file_uploader("Upload Insurance Policy Document 2", type=['txt', 'pdf'])

    user_question = st.text_input("Wat wil je graag weten?")

    if user_question:
       answer = process_document(selected_document_path, user_question)
       st.write(answer)
    
    
if __name__ == "__main__":
    main()