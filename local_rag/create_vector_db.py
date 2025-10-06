"""
- Put data into vector-enabled database
- Vector database hosted locally using ChromaDB
- When user asks a question, we look up relevant docs in the database
- Pass retrieved docs + original question to LLM
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

    def __init__(
        self,
        db_location: Path,
        embedding_model: str,
        collection_name: str,
        csv_path: Path,
    ):
        self.db_location = db_location
        self.embedding_model = embedding_model
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.collection_name = collection_name
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
        print(f"Count before: {vector_store._collection.count()}")
        vector_store.add_documents(documents=documents, ids=ids)
        print(f"Count after: {vector_store._collection.count()}")

    def add_to_existing_collection(self):
        """
        We are adding data to an existing DB.
        We need to find out how many documents are already in the DB: call this n.
        When adding the new data, the IDs should start from n + 1 
        so that we don't overwrite existing data.
        """
        num_docs = VectorDBReader.get_num_docs_in_collection(self.embedding_model, self.db_location, self.collection_name)
        documents, ids = self.load_data()
        new_ids = [str(int(id) + num_docs) for id in ids]
        print(f"Adding data to {self.collection_name} collection")
        self.add_documents_to_vector_store(documents, new_ids)
        print("Complete")

    def create_new_collection(self):
        print(f"Adding data to new collection: {self.collection_name}")
        documents, ids = self.load_data()
        vector_store = self.add_documents_to_vector_store(documents, ids)
        print("Complete")

    def run(self):
        if self.db_location.exists():
            self.add_to_existing_collection()
        else:
            self.create_new_collection()
            

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

    def get_num_docs_in_collection(embedding_model: str, db_location: str, collection_name) -> int:
        embeddings = OllamaEmbeddings(model=embedding_model)
        vectordb = Chroma(collection_name=collection_name, persist_directory=db_location, embedding_function=embeddings)
        return vectordb._collection.count()

    def get_collection_contents(embedding_model: str, db_location: str, collection_name) -> dict:
        embeddings = OllamaEmbeddings(model=embedding_model)
        vectordb = Chroma(collection_name=collection_name, persist_directory=db_location, embedding_function=embeddings)
        return vectordb._collection.get(include=["documents", "metadatas"])


class ValidateVectorDBVars(BaseModel):
    db_location: Path
    embedding_model: str
    collection_name: str
    csv_path: Path
    num_docs_to_retrieve: int

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

    csv_path = Path("/Users/ashapatel/Documents/projects/local-rag/data/new_data.csv")

    create_vector_db = VectorDBWriter(db_location, embedding_model, collection_name, csv_path)
    create_vector_db.run()

    VectorDBReader.list_all_collections(embedding_model, db_location)
    # result = VectorDBReader.get_collection_contents(embedding_model, db_location, collection_name)
    # print(result)



if __name__ == "__main__":
    main()
