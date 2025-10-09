import chromadb
from pathlib import Path
from config import Config
from data_models import ValidateVectorDBVars
from langchain_core.vectorstores import VectorStoreRetriever


config = Config()


class VectorDBCollectionUtils:
    """
    Utils functions that operate on the specified collection name.
    """

    def __init__(
        self,
        db_location: Path,
        embedding_model: str,
        collection_name: str
    ):
        self.db_location = db_location
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectordb_collection = Chroma(
            collection_name=self.collection_name, 
            persist_directory=self.db_location, 
            embedding_function=self.embeddings
        )

    def get_retriever(self, num_docs_to_retrieve: int) -> VectorStoreRetriever:
        """
        Make vector store retrievable by LLM.
        """
        retriever = self.vectordb_collection.as_retriever(
                search_kwargs = {"k": int(num_docs_to_retrieve)} 
            )
        return retriever

    def get_num_docs_in_collection(self) -> int:
        return self.vectordb_collection._collection.count()

    def get_collection_contents(self) -> dict:
        return self.vectordb_collection._collection.get(include=["documents", "metadatas"])


class VectorDBUtils:
    """
    Vector DB utils that are not specific to a collection.
    """

    def __init__(
        self,
        db_location: Path,
        embedding_model: str,
        csv_path: Path,
    ):
        self.db_location = db_location
        self.embedding_model = embedding_model
        self.embeddings = OllamaEmbeddings(model=embedding_model)

    def list_all_collections(self):
        vectordb_all_collections = Chroma(
            persist_directory=self.db_location, 
            embedding_function=self.embeddings
        )

        client = vectordb_all_collections._client
        collections = client.list_collections()

        print(f"Found {len(collections)} collection(s):")
        for collection in collections:
            print(f"  - Name: '{collection.name}'")
            print(f"    Documents: {collection.count()}")
            print()

    def delete_collection(self, collection_name: str):
        client = chromadb.PersistentClient(path=self.db_location)
        client.delete_collection(name=collection_name)


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

    vector_db_utils = VectorDBUtils(db_location, embedding_model, collection_name)
    vector_db_utils.get_num_docs_in_collection()


if __name__ == "__main__":
    main()