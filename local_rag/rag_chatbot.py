
"""
Main file to run in terminal to enable user to send questions to the RAG chatbot.
"""

from pathlib import Path
from dotenv import dotenv_values
from langchain_chroma import Chroma
from create_vector_db import CreateVectorDB
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from pydantic_core import PydanticCustomError
from pydantic import BaseModel, field_validator


config = dotenv_values("/Users/ashapatel/Documents/projects/local-rag/.env")


class ValidateArgs(BaseModel):
    db_location: Path
    embedding_model: str
    num_docs_to_retrieve: int
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
    chat_model = config["CHAT_MODEL"]
    prompt = config["BASIC_PROMPT"]

    ValidateArgs(
        db_location=db_location,
        embedding_model=embedding_model,
        num_docs_to_retrieve=num_docs_to_retrieve,
        chat_model=chat_model
    )

    return db_location, embedding_model, num_docs_to_retrieve, chat_model, prompt


def run():

    db_location, embedding_model, num_docs_to_retrieve, chat_model, prompt_template = load_args()

    model = OllamaLLM(model = chat_model)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model

    while True:
        question = input("Ask a question (q to quit): \n")
        if question == "q":
            break

        embeddings = OllamaEmbeddings(model=embedding_model)
        vectordb = Chroma(persist_directory=db_location, embedding_function=embeddings)
        
        retriever = vectordb.as_retriever(
                search_kwargs = {"k": int(num_docs_to_retrieve)} 
            )

        reviews = retriever.invoke(question)
        # retriever embeds question and searches for top k similar entries in the db

        result = chain.invoke(
            {
                "reviews": reviews, 
                "question": question
            }
        )
        print(result)


if __name__ == "__main__":
    run()