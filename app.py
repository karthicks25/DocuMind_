import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("<Your_API_Key>")
genai.configure(api_key=os.getenv("<Your_API_Key>"))

def get_pdf_text(pdfs):
    text = ""
    for doc in pdfs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text


def get_text_chunks(text):
    splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks= splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="<Your_API_Key>")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as Possible from peovided context make sure to provide all details,if the answer is not available
    in the provided context just say,"answer is not available in the context",don't provide the wrong answer\n\n
    context:\n{context}?\n
    question: \n{question}\n
    
    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro")

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(question):
    GOOGLE_API_KEY = "<Your_API_Key>"
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)

    new_db=FAISS.load_local("faiss_index",embeddings=embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(question)
    chain=get_conversational_chain()

    response=chain(
        {"input_documents":docs,"question":question}
        , return_only_outputs=True)
    print(response)
    st.write(response["output_text"])


def main(): 
    st.set_page_config("DocuMind")
    st.header("DocuMind")

    question = st.text_input("Ask question from uploaded PDFs")

    with st.sidebar:
        st.write("Menu:")
        pdf = st.file_uploader("Upload your PDF file...", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Executing..."):
                raw_text=get_pdf_text(pdf)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
    if question:
        user_input(question)   

main()
