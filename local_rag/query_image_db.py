import pandas as pd
from pathlib import Path
from config import Config

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

config = Config() 

db_location = Path(config.vector_db_path)
client = chromadb.PersistentClient(path=db_location)
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

def get_collection(collection_name: str) -> chromadb.Collection:
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        data_loader=data_loader
    )
    return collection


def main():

    collection_name = "image_text_collection"
    collection = get_collection(collection_name)

    result = collection.query(
        query_texts=["a machine learning engineer who is good at climbing"],
        n_results=1
    )

    print(result)



if __name__ == "__main__":
    main()