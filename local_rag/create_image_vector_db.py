"""
https://docs.trychroma.com/docs/embeddings/multimodal
"""

import pandas as pd
from pathlib import Path
from config import Config

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

config = Config()

class VectorDBImageWriter:

    def __init__(
        self,
        db_location: Path
    ):
        self.client = chromadb.PersistentClient(path=db_location)
        self.data_loader = ImageLoader()
        self.embedding_function = OpenCLIPEmbeddingFunction()

    def load_data(self, data_csv_path: Path) -> pd.DataFrame:
        return pd.read_csv(data_csv_path)

    def create_new_collection(self, collection_name: str):
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            data_loader=self.data_loader
        )

    def add_to_collection(self, data_csv_path: Path):
        data_df = self.load_data(data.csv_path)
        collection.add(
            ids=data_df[ids],
            uris=data_df["image path"],
            documents=data_df["image description"]
        )

    def list_collections(self):
        all_collections = self.client.list_collections()
        print("Available collections:")
        for col in all_collections:
            print(f"  - {col.name}")
            print(f"    - {col.count()}")


def main():

    db_location = Path(config.vector_db_path)

    image_writer = VectorDBImageWriter(db_location)
    image_writer.list_collections()



if __name__ == "__main__":
    main()