import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document



def extract_text_from_pdf(file):
    reader = PdfReader(file)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return pages_text

def load_and_process_document(file, user_question):
    # Extract text from the document
    document_pages = extract_text_from_pdf(file)
    
    # Create a list of Document objects from the text
    docs = [Document(page_content=page_text) for page_text in document_pages]

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Create question-answering chain
    llm = ChatOpenAI(model_name="gpt-4-turbo-preview", streaming=True)
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    # Perform similarity search and get relevant document chunks
    docs = [Document(page_content=str(text)) for text in texts]
    docs = vectorstore.similarity_search(user_question, k=len(docs))
    document_text = " ".join([doc.page_content for doc in docs])

    # Process the document and get the answer stream
    answer_stream = process_document(document_text, user_question, qa_chain)
    return answer_stream

def process_document(document_text, user_question, qa_chain):
    template = """
    Je bent een ervaren verzekeringsadviseur met diepgaande kennis van polisvoorwaarden. Jouw taak is om op betrouwbare en nauwkeurige wijze dekkingen, uitsluitingen en voorwaarden te analyseren en uit te leggen op basis van de tekst uit de geladen polisdocumenten. Het is essentieel dat je relevante informatie uit deze documenten rechtstreeks citeert en verwijst, inclusief pagina- of sectienummers indien beschikbaar.

    Als je een vraag tegenkomt waarbij de informatie in de documenten niet voldoende is om een volledige analyse te geven, vraag dan om verduidelijking over wat er gespecificeerd moet worden om een uitgebreid antwoord te kunnen geven. Voor vragen die gemakkelijk en rechtstreeks uit de tekst beantwoord kunnen worden, citeer en verwijs dan naar de relevante informatie.

    Bij het analyseren van de dekking, geef een duidelijke uitleg van eventuele specifieke voorwaarden of uitsluitingen die van toepassing zijn. Vermijd onnodige disclaimers, maar het is cruciaal om de voorwaarden voor dekking of uitsluiting nauwkeurig te vermelden en uit te leggen.

    Controleer altijd of er eventuele specifieke dekking limieten of maximale uitkeringen gelden voor de betreffende posten, en vermeld deze expliciet in je analyse. Het is van cruciaal belang dat deze informatie nauwkeurig wordt gepresenteerd en niet wordt verward met iets anders.

    Let goed op eventuele optionele dekkingen of modules die in het polisdocument worden vermeld, aangezien deze van grote invloed kunnen zijn op de dekkingsanalyse.

    Voor vragen over specifieke artikelen of scenario's, zoals wintersportuitrusting of gehuurde artikelen op een reisverzekering, zorg ervoor dat je duidelijk onderscheid maakt en ingaat op eventuele specifieke dekkingsvoorwaarden of uitsluitingen die van toepassing zijn.

    Gegeven de tekst uit de polisvoorwaarden: '{document_text}', en de vraag van de gebruiker: '{user_question}', hoe zou je de dekking analyseren en uitleggen op basis van de bovenstaande instructies?
    """

    prompt = ChatPromptTemplate.from_template(template)
    answer_stream = qa_chain({"input_documents": [document_text], "question": user_question, "prompt": prompt.format(document_text=document_text, user_question=user_question)})
    return answer_stream

def compare_documents(doc1_answer_stream, doc2_answer_stream, user_question):
    template = """
    Je bent een ervaren verzekeringsadviseur met de taak om de polisvoorwaarden van twee verschillende verzekeringsdocumenten te vergelijken om te bepalen of er verschillen zijn in de dekking voor een specifiek scenario.

    De analyse van de dekking van het eerste document met betrekking tot het scenario: '{doc1_answer}'

    De analyse van de dekking van het tweede document met betrekking tot het scenario: '{doc2_answer}'

    De oorspronkelijke vraag van de gebruiker was: '{user_question}'

    Op basis van de twee gegeven analyses, gebruik de Socratische methode om de geboden dekking van de twee documenten voor dit specifieke scenario kritisch te onderzoeken en te vergelijken. Ga een grondige en rigoureuze dialoog aan, stel prikkelende vragen om eventuele verschillen in dekking, uitsluitingen, voorwaarden of beperkingen tussen de twee documenten te identificeren. Als de dekking hetzelfde is, vermeld dat dan, maar zorg ervoor dat je conclusie wordt ondersteund door een uitgebreid onderzoek van alle relevante factoren.

    Geef een duidelijke en beknopte conclusie waarin je je analyse en vergelijking van de dekking van de twee documenten voor het gegeven scenario samenvat, en zorg ervoor dat je conclusie goed onderbouwd en ondersteund wordt door sterke bewijzen uit de analyses.
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Get the answer streams as strings
    doc1_answer = ''.join(doc1_answer_stream)
    doc2_answer = ''.join(doc2_answer_stream)

    # Perform comparison
    compare_prompt = """
    Je hebt de analyses van de dekkingen van twee verschillende polisdocumenten voor een specifiek scenario. Jouw taak is om een zeer grondige vergelijking te maken tussen deze twee analyses en een zo volledig mogelijke conclusie te trekken over eventuele verschillen of overeenkomsten in de dekking.

    Gebruik de Socratische methode en stel jezelf kritische vragen om alle aspecten van de dekking te onderzoeken, zoals:

    - Zijn er verschillen in de dekking voor het specifieke scenario?
    - Zijn er verschillen in uitsluitingen of voorwaarden?
    - Zijn er verschillen in dekking limieten of maximale uitkeringen?
    - Zijn er verschillen in optionele dekkingen of modules die van invloed zijn?
    - Zijn er andere factoren die van belang kunnen zijn voor de dekking in dit scenario?

    Ga diep in op alle relevante details uit de analyses en laat geen enkel aspect onbesproken. Formuleer een volledig onderbouwde en goed gemotiveerde conclusie waarin je alle belangrijke overeenkomsten en verschillen behandelt.

    Analyse document 1: '{doc1_answer}'

    Analyse document 2: '{doc2_answer}'

    Oorspronkelijke vraag van de gebruiker: '{user_question}'

    Geef nu een zeer grondige vergelijking en een volledig onderbouwde conclusie over de verschillen of overeenkomsten in de dekking tussen deze twee documenten voor het gegeven scenario.
    """

    compare_prompt = ChatPromptTemplate.from_template(compare_prompt)

    # Perform comparison
    llm = OpenAI(model_name="gpt-4", temperature=0, streaming=True)
    chain = compare_prompt | llm | StrOutputParser()
    return chain.stream({
        "doc1_answer": doc1_answer,
        "doc2_answer": doc2_answer,
        "user_question": user_question,
    })

def main():
    st.set_page_config(page_title="Vergelijker Polisvoorwaarden")
    st.title("Vergelijker Polisvoorwaarden")

    # File Upload
    uploaded_files = st.file_uploader("Upload Polisdocumenten", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        user_question = st.text_input("Voer je vraag over de polisdekking in:")

        if user_question:
            doc1_answer_stream = load_and_process_document(uploaded_files[0], user_question)
            doc2_answer_stream = load_and_process_document(uploaded_files[1], user_question)

            comparison_stream = compare_documents(doc1_answer_stream, doc2_answer_stream, user_question)
            st.write_stream(comparison_stream)

if __name__ == "__main__":
    main()