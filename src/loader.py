import os

from langchain_community.document_loaders import (
    PyPDFLoader
)


def load_pdf(folder_path):

    documents = []

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            path = os.path.join(
                folder_path,
                file
            )

            loader = PyPDFLoader(path)

            docs = loader.load()

            documents.extend(docs)

    return documents