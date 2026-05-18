from langchain_ollama import (
    OllamaEmbeddings
)


def load_embeddings():

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    return embeddings