from dataclasses import dataclass

@dataclass
class Config:
    text_data_csv_path: str = "/Users/ashapatel/Documents/projects/local-rag/data/text/pizza_reviews.csv"
    image_data_csv_path: str = "/Users/ashapatel/Documents/projects/local-rag/data/image_data_dict.csv"

    vector_db_path: str = "/Users/ashapatel/Documents/projects/local-rag/chroma_langchain_db"
    image_collection_name: str = "image_text_collection"
    num_results_to_retrieve: int = 3