import pandas as pd
from pathlib import Path
from config import Config

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from pydantic_core import PydanticCustomError
from pydantic import BaseModel, field_validator


config = Config() 
embedding_function = OpenCLIPEmbeddingFunction()


class ValidateArgs(BaseModel):
    db_location: Path
    num_results_to_retrieve: int
    collection_name: str

    @field_validator("db_location")
    def validate_db_exists(cls, value: Path):
        if not value.exists():
            raise PydanticCustomError(
                "db_doesn't_exist_error",
                "DB path doesn't exist.",
                {"path": value}
            )


def load_args():
    db_location = Path(config.vector_db_path)
    num_results_to_retrieve = config.num_results_to_retrieve
    collection_name = config.image_collection_name

    ValidateArgs(
        db_location=db_location,
        num_results_to_retrieve=num_results_to_retrieve,
        collection_name=collection_name,
    )

    return (
        db_location, 
        num_results_to_retrieve,
        collection_name,
    )


def get_collection(collection_name: str, client: chromadb.Client) -> chromadb.Collection:
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        data_loader=ImageLoader()
    )
    return collection


def run():
    db_location, num_results_to_retrieve, collection_name = load_args()
    client = chromadb.PersistentClient(path=db_location)

    while True:
        question = input("Describe the image you are looking for: \n")
        if question == "q":
            break
        
        collection = get_collection(collection_name, client)
        result = collection.query(
            query_texts=[question],
            n_results=num_results_to_retrieve
        )
        print(result["metadatas"])
    


if __name__ == "__main__":
    run()