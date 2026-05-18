from langchain.chains import (
    RetrievalQA
)

from langchain.prompts import (
    PromptTemplate
)


# ======================================
# CUSTOM RAG PROMPT
# ======================================

prompt_template = """

You are a helpful AI assistant.

Answer ONLY from the provided context.

Do NOT make up information.

If the answer is not present in the context,
say:

"I could not find the answer in the provided document."

Keep the answer factual, concise, and grounded.

Context:
{context}

Question:
{question}

Answer:

"""


PROMPT = PromptTemplate(

    template=prompt_template,

    input_variables=[
        "context",
        "question"
    ]
)


# ======================================
# RAG PIPELINE
# ======================================

def run_rag(
    query,
    retriever,
    llm
):

    qa_chain = RetrievalQA.from_chain_type(

        llm=llm,

        retriever=retriever,

        return_source_documents=True,

        chain_type_kwargs={
            "prompt": PROMPT
        }
    )

    result = qa_chain.invoke({
        "query": query
    })

    return {

        "answer": result["result"],

        "context": result[
            "source_documents"
        ]
    }