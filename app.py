import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import PyPDF2Loader
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def load_and_process_document(file_path, user_question):
    # Extract text from the document
    loader = PyPDF2Loader(file_path)
    pages = loader.load_and_split()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)

    # Create vector store
    embeddings = None  # Set to None for using OpenAI's pre-trained embeddings
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Create question-answering chain
    llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model_name="gpt-4-turbo-preview", streaming=True, temperature = 0, callbacks=[get_openai_callback()])
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    # Perform similarity search and get relevant document chunks
    docs = vectorstore.similarity_search(user_question)
    document_text = " ".join([doc.page_content for doc in docs])

    # Process the document and get the answer stream
    answer_stream = process_document(document_text, user_question, qa_chain)
    return answer_stream

def process_document(document_text, user_question, qa_chain):
    template = """
    You are an experienced insurance advisor with in-depth knowledge of policy terms and conditions. Your task is to reliably and accurately analyze and explain specific coverages, exclusions, and conditions based on the text from the loaded policy documents. It is essential that you directly quote and reference relevant information from these documents, including page numbers or section numbers if available.

    If you encounter a question where the information in the documents is not sufficient to provide a complete analysis, ask for clarification on what needs to be specified to provide a comprehensive answer. For questions that can be easily and directly answered from the text, quote and reference the relevant information directly.

    When analyzing coverage, provide a clear explanation of any specific conditions or exclusions that apply. While avoiding unnecessary disclaimers, it is crucial to accurately state and explain the conditions for coverage or exclusion.

    Additionally, always check if there are any specified coverage limits or maximum payouts for the items in question, and explicitly mention these in your analysis. It is critical that this information is accurately presented and not confused with anything else.

    Pay close attention to any optional coverages or modules that may be mentioned in the policy document, as these can significantly impact the coverage analysis.

    For questions about specific items or scenarios, such as winter sports equipment or hired items on a travel insurance policy, ensure that you clearly distinguish and address any specific coverage conditions or exclusions that apply.

    Given the text from the policy terms and conditions: '{document_text}', and the user's question: '{user_question}', how would you analyze and explain the coverage based on the above instructions?
    """

    prompt = ChatPromptTemplate.from_template(template)
    answer_stream = qa_chain(prompt=prompt, question=user_question, context=document_text)
    return answer_stream

def compare_documents(doc1_answer_stream, doc2_answer_stream, user_question):
    template = """
    You are an experienced insurance advisor tasked with comparing the policy terms and conditions of two different insurance documents to determine if there are any differences in coverage for a specific scenario.

    The analysis of the first document's coverage regarding the scenario: '{doc1_answer}'

    The analysis of the second document's coverage regarding the scenario: '{doc2_answer}'

    The original user question was: '{user_question}'

    Based on the two analyses provided, use the Socratic method to critically examine and compare the coverage offered by the two documents for this specific scenario. Engage in a thorough and rigorous dialogue, asking probing questions to identify any differences in coverage, exclusions, conditions, or limitations between the two documents. If the coverage is the same, state that as well, but ensure that your conclusion is supported by a comprehensive examination of all relevant factors.

    Provide a clear and concise conclusion summarizing your analysis and comparison of the two documents' coverage for the given scenario, ensuring that your conclusion is well-reasoned and supported by strong evidence from the analyses.
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Get the answer streams as strings
    doc1_answer = ''.join(doc1_answer_stream)
    doc2_answer = ''.join(doc2_answer_stream)

    # Perform comparison
    llm = OpenAI(model_name="gpt-4", temperature=0, streaming=True)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "doc1_answer": doc1_answer,
        "doc2_answer": doc2_answer,
        "user_question": user_question,
    })

def main():
    st.title("Policy Document Comparison Tool")

    # File Upload
    uploaded_files = st.file_uploader("Upload Policy Documents", accept_multiple_files=True)

    if uploaded_files:
        user_question = st.text_input("Enter your question about the policy coverage:")

        if user_question:
            doc1_answer_stream = load_and_process_document(uploaded_files[0], user_question)
            doc2_answer_stream = load_and_process_document(uploaded_files[1], user_question)

            comparison_stream = compare_documents(doc1_answer_stream, doc2_answer_stream, user_question)
            st.write_stream(comparison_stream)

if __name__ == "__main__":
    main()