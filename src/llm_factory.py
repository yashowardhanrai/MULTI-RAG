from langchain_ollama import (
    ChatOllama
)


MODELS = {

    "llama": "llama3.1:8b",

     "mistral": "mistral:7b",

     "gemma": "gemma3:4b",

     "phi3": "phi3:mini"
}


def load_llm(model_key):

    model_name = MODELS[model_key]

    llm = ChatOllama(
        model=model_name,
        temperature=0
    )

    return llm