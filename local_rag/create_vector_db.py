"""
- Put data into vector-enabled database
- Vector database hosted locally using ChromaDB
- When user asks a question, we look up relevant docs in the database
- Pass retrieved docs + original question to LLM

NOTE: currently this code only allows you to create a DB from scratch,
not to add to an existing DB. The db_location must not already exist.

I WANT to be able to:
- load the retriever for an existing collection
- add to an existing collection
- create & add to a new collection
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

from pydantic_core import PydanticCustomError
from pydantic import BaseModel, field_validator

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


config = dotenv_values("/Users/ashapatel/Documents/projects/local-rag/.env")


class VectorDBWriter:

    """
    TO DO: enable writing to an existing collection
    """

    def __init__(
        self,
        db_location: Path,
        embedding_model: str,
        collection_name: str,
        csv_path: Path,
        create_new_db: bool
    ):
        self.db_location = db_location
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.collection_name = collection_name
        self.create_new_db = create_new_db
        self.csv_path = csv_path

    def load_data(self) -> tuple[list, list]:
        """
        Load csv data into a list of documents and ids.
        This format is required by the vector data base.
        """
        data_df = pd.read_csv(self.csv_path)
        documents = []
        ids = []
        for i, row in data_df.iterrows():
            document = Document(
                page_content=row["Title"] + " " + row["Review"],
                metadata = {"rating": row["Rating"], "date": row["Date"]},
                id = str(i)
            )
            ids.append(str(i))
            documents.append(document)

        return documents, ids
    
    def add_documents_to_vector_store(self, documents: list, ids: list) -> Chroma:
        """
        Initialise the vector store and add the embedded data.
        """

        vector_store = Chroma(
            collection_name = self.collection_name,
            persist_directory = self.db_location,
            embedding_function = self.embeddings
        )
        vector_store.add_documents(documents=documents, ids=ids)

    def run(self):
        if self.create_new_db:               
            print("Adding data to vector DB...")
            documents, ids = self.load_data()
            vector_store = self.add_documents_to_vector_store(documents, ids)
            print("Complete")
        # else:
            ## figure out how to write to an existing collection


class VectorDBReader:

    def get_retriever(embedding_model: str, db_location: str, num_docs_to_retrieve: int, collection_name: str) -> VectorStoreRetriever:
        """
        Make vector store retrievable by LLM.
        """
        embeddings = OllamaEmbeddings(model=embedding_model)
        vectordb = Chroma(collection_name=collection_name, persist_directory=db_location, embedding_function=embeddings)
        retriever = vectordb.as_retriever(
                search_kwargs = {"k": int(num_docs_to_retrieve)} 
            )
        return retriever

    def list_all_collections(embedding_model: str, db_location: str):
            embeddings = OllamaEmbeddings(model=embedding_model)
            vectordb = Chroma(
                persist_directory=db_location, 
                embedding_function=embeddings
            )

            client = vectordb._client
            collections = client.list_collections()

            print(f"Found {len(collections)} collection(s):")
            for collection in collections:
                print(f"  - Name: '{collection.name}'")
                print(f"    Documents: {collection.count()}")
                print()


class ValidateVectorDBVars(BaseModel):
    db_location: Path
    embedding_model: str
    collection_name: str
    csv_path: Path
    num_docs_to_retrieve: int

    @field_validator("db_location")
    def validate_db_location(cls, value: Path):
        if value.exists():
            raise PydanticCustomError(
                "path_already_exists_error",
                "Vector DB path already exists.",
                {"path": value}
            )

    @field_validator("csv_path")
    def validate_csv_path(cls, value: Path):
        if not value.exists():
            raise PydanticCustomError(
                "path_doesn't_exist_error",
                "CSV path doesn't exist.",
                {"path": value}
            )


def main():

    db_location = Path(config["VECTOR_DB_PATH"])
    embedding_model = config["EMBEDDING_MODEL"]
    collection_name = config["COLLECTION_NAME"]
    csv_path = Path(config["DATA_CSV_PATH"])
    num_docs_to_retrieve = int(config["NUM_DOCS_TO_RETRIEVE"])

    ValidateVectorDBVars(
        db_location=db_location,
        embedding_model=embedding_model,
        collection_name=collection_name,
        csv_path=csv_path,
        num_docs_to_retrieve=num_docs_to_retrieve
    )

    create_new_db = True

    create_vector_db = VectorDBWriter(db_location, embedding_model, collection_name, csv_path, create_new_db)
    create_vector_db.run()


if __name__ == "__main__":
    main()
