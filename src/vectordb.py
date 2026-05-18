import os

from langchain_chroma import Chroma


DB_DIR = "chroma_db"


def create_vectorstore(
    chunks,
    embeddings
):

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    return vectorstore


def load_vectorstore(embeddings):

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    return vectorstore


def vectorstore_exists():

    return os.path.exists(DB_DIR)