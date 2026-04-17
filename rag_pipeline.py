import os
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from typing import List

load_dotenv()

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Storage for chunks and their text
chunk_texts = []
faiss_index = None

def build_index(docs_folder="docs"):
    global faiss_index, chunk_texts

    print("Loading documents...")
    docs = SimpleDirectoryReader(docs_folder).load_data()

    print("Chunking documents...")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(docs)

    print(f"Total chunks: {len(nodes)}")

    # Embed each chunk
    texts = [node.get_content() for node in nodes]
    chunk_texts = texts

    print("Generating embeddings...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    print("Index built successfully!")
    return faiss_index, chunk_texts

def retrieve(query, top_k=3) -> List:
    global faiss_index, chunk_texts

    if faiss_index is None:
        build_index()

    query_embedding = embedder.encode([query]).astype("float32")
    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        if i < len(chunk_texts):
            results.append(chunk_texts[i])

    return results

if __name__ == "__main__":
    build_index()
    print("\nTest retrieval:")
    results = retrieve("caller feels numb and disconnected")
    for i, r in enumerate(results):
        print(f"\n--- Chunk {i+1} ---\n{r[:300]}")