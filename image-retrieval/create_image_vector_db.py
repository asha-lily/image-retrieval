"""
https://docs.trychroma.com/docs/embeddings/multimodal

Create / add to an image database (chromadb 'collection')
"""

import pandas as pd
from pathlib import Path
from config import Config

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

config = Config()


def load_csv_data(data_csv_path: Path) -> tuple[list, list, list]:
    data_df = pd.read_csv(data_csv_path)
    ids = [str(id) for id in list(data_df["id"])]
    image_paths = [str(image_path) for image_path in list(data_df["image path"])]
    descriptions = [str(description) for description in list(data_df["image description"])]
    return ids, image_paths, descriptions


class VectorDBImageWriter:

    def __init__(
        self,
        db_location: Path
    ):
        self.client = chromadb.PersistentClient(path=db_location)
        self.data_loader = ImageLoader()
        self.embedding_function = OpenCLIPEmbeddingFunction()

    def create_new_collection(self, collection_name: str):
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            data_loader=self.data_loader
        )

    def add_to_collection(self, data_csv_path: Path, collection_name: str):
        ids, image_paths, descriptions = load_csv_data(data_csv_path)
        text_metadata = [{"text": description} for description in descriptions]

        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            data_loader=self.data_loader
        )

        collection.add(
            ids=ids,
            uris=image_paths,
            metadatas=text_metadata
        )

    def list_collections(self):
        all_collections = self.client.list_collections()
        print("Available collections:")
        for col in all_collections:
            print(f"  - {col.name}")
            print(f"    - {col.count()}")

    def view_collection_contents(self, collection_name: str):
        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            data_loader=self.data_loader
        )

        collection_contents = collection.get(include=['uris', 'metadatas'])
        print("\n--- All items ---")
        print(f"Number of items: {len(collection_contents['ids'])}")
        print("IDs:", collection_contents['ids'])
        print("URIs:", collection_contents['uris'])
        print("Metadatas:", collection_contents['metadatas'])


def main():

    db_location = Path(config.vector_db_path)
    data_csv_path = config.image_data_csv_path

    collection_name = "image_text_collection"

    image_writer = VectorDBImageWriter(db_location)
    # image_writer.add_to_collection(data_csv_path, collection_name)
    image_writer.view_collection_contents(collection_name)

    # image_writer.list_collections()



if __name__ == "__main__":
    main()