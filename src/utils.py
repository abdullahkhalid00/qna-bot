from pymongo import MongoClient
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA


def load_data(
        dir_path: str,
        glob: str
) -> list:
    loader = DirectoryLoader(
        dir_path,
        glob=glob,
        show_progress=True
    )
    return loader.load()

def load_mongodb_collection(
        client_url: str,
        db_name: str,
        collection_name: str
):
    client = MongoClient(client_url)
    return client[db_name][collection_name]

def split_documents(documents: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "],
        length_function=len
    )
    return splitter.split_documents(documents)

def create_vector_store(
        documents: list,
        collection,
        embeddings,
        index_name: str
):
    MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=index_name
    )

def load_vector_store(
        collection,
        embeddings
):
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings
    )
    return vector_store

def create_retrieval_chain(llm, vector_store):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa
