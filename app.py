import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.document_comparators import CompareDocuments, CompareDocumentsSummary
from langchain.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

def extract_text_from_pdf_by_page(file_path):
    pages_text = []
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return pages_text

def process_document(document_path):
    loader = PyPDFLoader(document_path)
    return loader.load()

def compare_documents(doc1, doc2, user_question):
    compare_docs = CompareDocuments()
    comparison_summary = CompareDocumentsSummary()

    prompt_template = """
    Given the following documents:
    
    Document 1: {doc_1}
    Document 2: {doc_2}

    And the user's question: {question}

    Provide a detailed comparison of the two documents, highlighting the similarities and differences in coverage, exclusions, conditions, and limitations related to the user's question. Use the Socratic method to thoroughly investigate all relevant factors, and provide a well-reasoned conclusion summarizing the key differences and similarities in coverage between the two documents for the given scenario.
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    comparison_result = compare_docs(doc1, doc2, prompt=prompt)
    summary = comparison_summary(comparison_result, question=user_question)

    llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "doc_1": doc1.page_content,
        "doc_2": doc2.page_content,
        "question": user_question
    })

def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([os.path.join(category_path, doc) for doc in os.listdir(category_path) if doc.endswith('.pdf')])

def display_search_results(search_results):
    if not search_results:
        st.write("Geen documenten gevonden.")
        return

    selected_title1 = st.selectbox("Selecteer het eerste document:", search_results, index=0)
    selected_title2 = st.selectbox("Selecteer het tweede document:", search_results, index=1)

    user_question = st.text_input("Stel een vraag over de polisvoorwaarden:")
    if user_question:
        document1 = process_document(selected_title1)
        document2 = process_document(selected_title2)
        comparison_stream = compare_documents(document1, document2, user_question)
        st.write_stream(comparison_stream)

def main():
    st.title("Polisvoorwaarden Vergelijker")
    categories = sorted(next(os.walk(BASE_DIR))[1])
    selected_category = st.selectbox("Kies een categorie:", categories)
    document_paths = get_documents(selected_category)
    display_search_results(document_paths)

if __name__ == "__main__":
    main()