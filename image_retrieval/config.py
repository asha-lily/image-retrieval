from dataclasses import dataclass

@dataclass
class Config:
    text_data_csv_path: str = "/Users/ashapatel/Documents/projects/local-rag/data/text/pizza_reviews.csv"
    image_data_csv_path: str = "/Users/ashapatel/Documents/projects/local-rag/data/image_data_dict.csv"

    vector_db_path: str = "/Users/ashapatel/Documents/projects/image-retrieval/chroma_langchain_db"
    num_results_to_retrieve: int = 3