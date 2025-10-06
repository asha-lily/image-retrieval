from dataclasses import dataclass

@dataclass
class Config:

    data_csv_path: str = "/Users/ashapatel/Documents/projects/local-rag/data/pizza_reviews.csv"

    vector_db_path: str = "./chroma_langchain_db"
    collection_name: str = "pizza_reviews"
    num_docs_to_retrieve: int = 5

    embedding_model: str = "mxbai-embed-large"
    chat_model: str = "llama3.2"
    model_provider: str = "ollama"

    basic_prompt: str = "You are a pizza connoisseur.\\Here are some reviews of pizza restaurants: {reviews}.\\Here is a question you need to answer: {question}"
