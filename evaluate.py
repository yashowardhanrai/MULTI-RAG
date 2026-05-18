import pandas as pd

from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy
)

from ragas.embeddings import (
    LangchainEmbeddingsWrapper
)

from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama

from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import load_embeddings

from src.vectordb import (
    create_vectorstore,
    load_vectorstore,
    vectorstore_exists
)

from src.retriever import create_retriever

from src.llm_factory import (
    load_llm,
    MODELS
)

from src.rag_pipeline import run_rag

from evaluation_dataset import (
    evaluation_data
)


# ======================================
# Evaluator LLM
# ======================================

evaluator_llm = ChatOllama(
    model="phi3:mini",
    temperature=0,
    format="json"
)


# ======================================
# RAGAS CONFIG
# ======================================

run_config = RunConfig(
    timeout=900,
    max_workers=1,
    max_wait=120
)


# ======================================
# LOAD EMBEDDINGS
# ======================================

print("\nLoading embeddings...\n")

embeddings = load_embeddings()

ragas_embeddings = (
    LangchainEmbeddingsWrapper(
        embeddings
    )
)


# ======================================
# LOAD / CREATE VECTORSTORE
# ======================================

if vectorstore_exists():

    print("Loading existing Chroma DB...\n")

    vectorstore = load_vectorstore(
        embeddings
    )

else:

    print("Creating new Chroma DB...\n")

    documents = load_pdf("data")

    print(f"\nLoaded Documents: {len(documents)}\n")

    chunks = split_documents(
        documents
    )

    print(f"\nChunks Created: {len(chunks)}\n")

    vectorstore = create_vectorstore(
        chunks,
        embeddings
    )


# ======================================
# CREATE RETRIEVER
# ======================================

retriever = create_retriever(
    vectorstore
)


# ======================================
# TEST RETRIEVAL
# ======================================

print("\n======================")
print("TEST RETRIEVAL")
print("======================\n")

test_docs = retriever.invoke(
    "What is this document about?"
)

for i, doc in enumerate(test_docs):

    print(f"\nRetrieved Chunk {i+1}:\n")

    print(doc.page_content[:500])

    print("\n" + "=" * 50)


# ======================================
# STORE RESULTS
# ======================================

all_results = []


# ======================================
# LOOP THROUGH MODELS
# ======================================

for model_key, model_name in MODELS.items():

    print("\n======================")
    print(f"MODEL: {model_name}")
    print("======================\n")

    llm = load_llm(model_key)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for sample in evaluation_data:

        question = sample["question"]

        ground_truth = sample[
            "ground_truth"
        ]

        print(f"\nQuestion: {question}")

        result = run_rag(
            query=question,
            retriever=retriever,
            llm=llm
        )

        answer = result["answer"]

        retrieved_contexts = []

        for doc in result["context"]:

            retrieved_contexts.append(
                str(doc.page_content)
            )

        print(f"\nAnswer:\n{answer}\n")

        questions.append(
            str(question)
        )

        answers.append(
            str(answer)
        )

        contexts.append(
            retrieved_contexts
        )

        ground_truths.append(
            str(ground_truth)
        )

    # ======================================
    # CREATE DATASET
    # ======================================

    data_samples = []

    for q, a, c, g in zip(
        questions,
        answers,
        contexts,
        ground_truths
    ):

        data_samples.append({

            "question": str(q),

            "answer": str(a),

            "contexts": [
                str(x) for x in c
            ],

            "ground_truth": str(g)
        })

    dataset = Dataset.from_list(
        data_samples
    )

    # ======================================
    # RUN RAGAS
    # ======================================

    print(
        f"\nRunning RAGAS for {model_name}\n"
    )

    try:

        scores = evaluate(

            dataset=dataset,

            metrics=[

                Faithfulness(),

                AnswerRelevancy()
            ],

            llm=evaluator_llm,

            embeddings=ragas_embeddings,

            run_config=run_config
        )

        scores_df = scores.to_pandas()

        print("\nRAW SCORES:\n")

        print(scores_df)

        avg_scores = scores_df.mean(
            numeric_only=True
        )

        # safer extraction

        faithfulness_score = avg_scores.get(
            "faithfulness",
            None
        )

        answer_rel_score = avg_scores.get(
            "answer_relevancy",
            None
        )

        model_result = {

            "model": model_name,

            "faithfulness":

                round(float(faithfulness_score), 4)

                if faithfulness_score is not None

                else None,

            "answer_relevancy":

                round(float(answer_rel_score), 4)

                if answer_rel_score is not None

                else None
        }

    except Exception as e:

        print(
            f"\nRAGAS FAILED for {model_name}\n"
        )

        print(e)

        model_result = {

            "model": model_name,

            "faithfulness": None,

            "answer_relevancy": None
        }

    all_results.append(
        model_result
    )

    print("\nAverage Scores:\n")

    print(model_result)


# ======================================
# FINAL RESULTS
# ======================================

final_df = pd.DataFrame(
    all_results
)

print("\n======================")
print("FINAL COMPARISON")
print("======================\n")

print(final_df)

final_df.to_csv(
    "results/ragas_results.csv",
    index=False
)

print(
    "\nResults saved successfully.\n"
)