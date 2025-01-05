import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from io import BytesIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ''
    if pdf_docs:
        if isinstance(pdf_docs, list):
            for pdf in pdf_docs:
                pdf_reader = PdfReader(BytesIO(pdf.read()))
                for page in pdf_reader.pages:
                    text += page.extract_text()
        else:  # Single file case
            pdf_reader = PdfReader(BytesIO(pdf_docs.read()))
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save the FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    os.makedirs('faiss-index', exist_ok=True)  # Ensure directory exists
    vector_store.save_local('faiss-index')

# Function to create the conversational chain
def get_conversational_chain():
    prompt_template = '''
    Answer the question as detailed as possible from the provided context. If the answer is not in the
    provided context, say "Answer is not available in the context" and do not provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    '''
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

# Function to handle user queries
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    try:
        if not os.path.exists('faiss-index/index.faiss'):
            raise FileNotFoundError("Vector store not found. Please process PDFs first.")
        new_db = FAISS.load_local('faiss-index', embeddings, allow_dangerous_deserialization=True)
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({'input_documents': docs, 'question': user_question})
    st.write('Reply: ', response['output_text'])

# Main menu and UI
def menu():
    st.set_page_config('Search The PDF')
    st.header('SmartPDF: Gemini-Driven Contextual Question-Answering')

    user_question = st.text_input('Ask a question from the PDF files')

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title('Menu:')
        pdf_docs = st.file_uploader('Upload your PDF file and Click on the Submit & Process Button')
        if st.button('Submit & Process'):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success('Vector store created successfully!')
                else:
                    st.error('No text found in the uploaded PDF.')
if __name__ == "__main__":
    menu()
