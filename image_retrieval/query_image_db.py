"""
Run this file to query the image DB from the command line.
"""

import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
from config import Config

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from schemas import ValidateQueryArgs


config = Config() 
embedding_function = OpenCLIPEmbeddingFunction()


def load_args():
    db_location = Path(config.vector_db_path)
    num_results_to_retrieve = config.num_results_to_retrieve

    runtime_args = parse_args()
    collection_name = runtime_args.collection_name

    ValidateQueryArgs(
        db_location=db_location,
        collection_name=collection_name,
        num_results_to_retrieve=num_results_to_retrieve,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection_name",
        type=str,
        default="default",
        help="The name of the collection you want to query."
    )
    return parser.parse_args()


def run():

    db_location, num_results_to_retrieve, collection_name = load_args()
    client = chromadb.PersistentClient(path=db_location)

    while True:
        question = input("Describe the image you are looking for (q to quit): \n")
        if question == "q":
            break
        
        collection = get_collection(collection_name, client)
        result = collection.query(
            query_texts=[question],
            n_results=num_results_to_retrieve,
            include=['metadatas', 'documents', 'distances', 'uris']
        )
        most_similar_image__path = result["uris"][0][0]
        most_similar_image_label = result["metadatas"][0][0]["text"]

        print(f"Found an image with label: {most_similar_image_label}")
        img = Image.open(most_similar_image__path
        )
        img.show()


if __name__ == "__main__":
    run()