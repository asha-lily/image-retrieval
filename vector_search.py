"""
- Put data into vector-enabled database
- Vector database hosted locally using ChromaDB
- When user asks a question, we look up relevant docs in the database
- Pass retrieved docs + original question to LLM
"""

import os
import pandas as pd
from dotenv import dotenv_values

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


config = dotenv_values(".env")


df = pd.read_csv("pizza_reviews.csv")
embeddings = OllamaEmbeddings(model = config["EMBEDDING_MODEL"])

db_location = "./chroma_langchain_db"
add_docs_to_db = not os.path.exists(db_location) 

# if db doesn't already exist, add the data
if add_docs_to_db:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata = {"rating": row["Rating"], "date": row["Date"]},
            id = str(i)
        )

        ids.append(str(i))
        documents.append(document)

# initialise the vector store
vector_store = Chroma(
    collection_name = "pizza_reviews",
    persist_directory = db_location,
    embedding_function = embeddings
)

# add embedded docs to vector store
if add_docs_to_db:
    vector_store.add_documents(documents=documents, ids=ids)

# make vector store retrievable by LLM
retriever = vector_store.as_retriever(
    search_kwargs = {"k": 5} # num docs to lookup
)