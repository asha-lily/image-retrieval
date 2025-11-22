from dataclasses import dataclass

@dataclass
class Config:
    image_data_csv_path: str = "/Users/ashapatel/Documents/projects/image-retrieval/image_retrieval/data/image_data_dict.csv"
    vector_db_path: str = "/Users/ashapatel/Documents/projects/image-retrieval/chroma_langchain_db"
    num_results_to_retrieve: int = 3