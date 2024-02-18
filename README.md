# QnA Bot 🤖 using RAG Pipeline

A simple question answering application built using `Gradio`, `MongoDB`, `LangChain` and `OpenAI` that lets the user question answer their own documents.

## Usage

Navigate to the project directory and install the required dependencies.

```bash
pip install -r requirements.txt
```

Use the sample data in the `./data` directory or use your own documents by pasting them in the same directory.

**Note:** Make sure to use `.txt` files.

Set up your own `OpenAI API KEY` and `MongoDB URI` in a new `.env` file using the provided `.env-sample` file as reference.

Run the `vector_store.py` script to create emebddings of your documents and store them on your remote MongoDB cluster.

### Setting up Atlas Vector Search

Create a new Atlas vector search index with the `ATLAS_VECTOR_SEARCH_INDEX_NAME` as its name. Select your `DB_NAME` and `COLLECTION_NAME` in the left sidebar and use the following `JSON` mappings.

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 1536,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
```

Wait for the search index to become active and run the `main.py` script to start chatting.
