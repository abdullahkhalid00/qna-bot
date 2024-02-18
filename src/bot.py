import os

from dotenv import load_dotenv
from utils import (
    load_mongodb_collection,
    load_vector_store,
    create_retrieval_chain
)
from langchain_openai import OpenAIEmbeddings, OpenAI
import gradio as gr


load_dotenv("../.env")


class Bot:
    def __init__(
            self,
            name: str,
            description: str
    ):
        self.name = name
        self.description = description

    def handle_user_query(self, query: str) -> str:
        collection = load_mongodb_collection(
            client_url=os.getenv("MONGODB_ATLAS_CLUSTER_URI"),
            db_name=os.getenv("DB_NAME"),
            collection_name=os.getenv("COLLECTION_NAME")
        )
        vector_store = load_vector_store(
            collection=collection,
            embeddings=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        )
        qa = create_retrieval_chain(
            llm=OpenAI(
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            ),
            vector_store=vector_store
        )
        response = qa.run(query)
        return response.strip()

    def display_interface(self):
        app = gr.Interface(
            title=self.name,
            description=self.description,
            fn=self.handle_user_query,
            inputs=gr.Textbox(label="Query"),
            outputs=gr.Textbox(label="Response"),
            theme=gr.themes.Monochrome(
                font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"]
            )
        )
        app.launch()
