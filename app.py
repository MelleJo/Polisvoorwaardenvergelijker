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
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def extract_text_from_pdf_by_page(uploaded_file):
    pages_text = []
    reader = PdfReader(uploaded_file)
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
        Jij bent expert verzekeringsacceptant, dus jij hebt diepgaande kennis en ervaring met het vergelijken van polisvoorwaarden. 
        De gebruiker stelt een vraag over de verschillen of overeenkomsten of welke voorwaarden het best zijn in scenario x. 

        Jij analyseert eerst de vraag van de gebruiker goed, maar dit doe je op de achtergrond en laat je niet aan de gebruiker zien.
        Hier is de vraag van de gebruiker: '{user_question}'.
        Dan vergelijk je de inhoud van de opgehaalde pagina's van de polisvoorwaarden {document_text_doc1} en {document_text_doc2} met elkaar.
        En beantwoord je de vraag zo nauwkeurig mogelijk. De gebruiker is iemand die werkt voor een verzekerings intermediar, en jij helpt met de juiste adviezen geven d.m.v. het versnellen van de vergelijkingen. 
    
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

    if doc1 and doc2 and user_question:
       selected_document_path = doc1 and doc2
       answer = process_document(selected_document_path, user_question)
       st.write(answer)
    
    
if __name__ == "__main__":
    main()