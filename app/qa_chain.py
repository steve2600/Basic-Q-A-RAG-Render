from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Load environment variables from .env in the same folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))



def get_qa_chain(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build vector DB from docs
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192",
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa
