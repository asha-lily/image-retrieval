
"""
Main file to run in terminal to enable user to send questions to the RAG chatbot.
"""

from pathlib import Path
from config import Config
from langchain_chroma import Chroma
from create_vector_db import VectorDBReader
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from pydantic_core import PydanticCustomError
from pydantic import BaseModel, field_validator


config = Config()


class ValidateArgs(BaseModel):
    db_location: Path
    embedding_model: str
    num_docs_to_retrieve: int
    collection_name: str
    chat_model: str

    @field_validator("db_location")
    def validate_db_exists(cls, value: Path):
        if not value.exists():
            raise PydanticCustomError(
                "db_doesn't_exist_error",
                "DB path doesn't exist.",
                {"path": value}
            )


def load_args():
    db_location = Path(config["VECTOR_DB_PATH"])
    embedding_model = config["EMBEDDING_MODEL"]
    num_docs_to_retrieve = int(config["NUM_DOCS_TO_RETRIEVE"])
    collection_name = config["COLLECTION_NAME"]
    chat_model = config["CHAT_MODEL"]
    prompt = config["BASIC_PROMPT"]

    ValidateArgs(
        db_location=db_location,
        embedding_model=embedding_model,
        num_docs_to_retrieve=num_docs_to_retrieve,
        collection_name=collection_name,
        chat_model=chat_model
    )

    return db_location, embedding_model, num_docs_to_retrieve, collection_name, chat_model, prompt


def run():

    db_location, embedding_model, num_docs_to_retrieve, collection_name, chat_model, prompt_template = load_args()

    model = OllamaLLM(model = chat_model)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model

    while True:
        question = input("Ask a question (q to quit): \n")
        if question == "q":
            break

        embeddings = OllamaEmbeddings(model=embedding_model)
        vectordb = Chroma(persist_directory=db_location, embedding_function=embeddings)
    
        # retriever embeds question and searches for top k similar entries in the db
        retriever = VectorDBReader.get_retriever(embedding_model, db_location, num_docs_to_retrieve, collection_name)

        reviews = retriever.invoke(question)

        result = chain.invoke(
            {
                "reviews": reviews, 
                "question": question
            }
        )

        print(result)


if __name__ == "__main__":
    run()