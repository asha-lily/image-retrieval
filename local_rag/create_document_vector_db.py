"""
- Put data into vector-enabled database
- Vector database hosted locally using ChromaDB
- When user asks a question, we look up relevant docs in the database
- Pass retrieved docs + original question to LLM
"""

import os
import chromadb
import pandas as pd
from pathlib import Path
from config import Config

from data_models import ValidateVectorDBVars

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from utils import VectorDBCollectionUtils


config = Config()


class VectorDBDocumentWriter:

    def __init__(
        self,
        db_location: Path,
        embedding_model: str,
        csv_path: Path,
    ):
        self.db_location = db_location
        self.embedding_model = embedding_model
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.csv_path = csv_path

    def load_csv_data(self) -> tuple[list, list]:
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
    
    def add_documents_to_vector_store(self, documents: list, ids: list, collection_name: str) -> Chroma:
        """
        Initialise the vector store and add the embedded data.
        """
        vector_store = Chroma(
            collection_name = collection_name,
            persist_directory = self.db_location,
            embedding_function = self.embeddings
        )
        print(f"Count before: {vector_store._collection.count()}")
        vector_store.add_documents(documents=documents, ids=ids)
        print(f"Count after: {vector_store._collection.count()}")

    def add_to_existing_collection(self, collection_name: str):
        """
        We are adding data to an existing DB.
        We need to find out how many documents are already in the DB: call this n.
        When adding the new data, the IDs should start from n + 1 
        so that we don't overwrite existing data.
        """
        num_docs = VectorDBCollectionUtils(self.db_location, self.embedding_model, self.csv_path).get_num_docs_in_collection()
        documents, ids = self.load_csv_data()
        new_ids = [str(int(id) + num_docs) for id in ids]
        print(f"Adding data to {self.collection_name} collection")
        self.add_documents_to_vector_store(documents, new_ids)
        print("Complete")

    def create_new_collection(self, collection_name: str):
        print(f"Adding data to new collection: {collection_name}")
        documents, ids = self.load_data()
        vector_store = self.add_documents_to_vector_store(documents, ids)
        print("Complete")

    def run(self, collection_name: str):
        if self.db_location.exists():
            self.add_to_existing_collection()
        else:
            self.create_new_collection()


def main():

    db_location = Path(config.vector_db_path)
    embedding_model = config.embedding_model
    collection_name = config.collection_name
    csv_path = Path(config.data_csv_path)
    num_docs_to_retrieve = config.num_docs_to_retrieve

    ValidateVectorDBVars(
        db_location=db_location,
        embedding_model=embedding_model,
        collection_name=collection_name,
        csv_path=csv_path,
        num_docs_to_retrieve=num_docs_to_retrieve
    )


if __name__ == "__main__":
    main()
