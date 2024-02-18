import os

from dotenv import load_dotenv
from utils import (
    load_data,
    load_mongodb_collection,
    split_documents,
    create_vector_store
)
from langchain_openai import OpenAIEmbeddings

load_dotenv(".env")


def main():
    try:
        docs = load_data("./data")
        collection = load_mongodb_collection(
            client_url=os.getenv("MONGODB_ATLAS_CLUSTER_URI"),
            db_name=os.getenv("DB_NAME"),
            collection_name=os.getenv("COLLECTION_NAME")
        )
        chunks = split_documents(docs)
        vector_store = create_vector_store(
            documents=chunks,
            collection=collection,
            embeddings=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
            index_name=os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")
        )
    except Exception as e:
        print(f"Error in creating vector store: {str(e)}")


if __name__ == "__main__":
    main()
