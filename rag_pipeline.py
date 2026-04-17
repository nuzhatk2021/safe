import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from typing import List, Tuple, Optional

load_dotenv()

# ---- Types ----
FaissIndex = faiss.IndexFlatL2
EmbeddingArray = np.ndarray

# ---- Globals ----
embedder: SentenceTransformer = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

chunk_texts: List[str] = []
faiss_index: Optional[FaissIndex] = None

def build_index(docs_folder: str | Path = "docs") -> Tuple[FaissIndex, List[str]]:
    global faiss_index, chunk_texts

    docs_path: Path = Path(docs_folder)

    print("Loading documents...")
    docs = SimpleDirectoryReader(str(docs_path)).load_data()

    print("Chunking documents...")
    splitter: SentenceSplitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )
    nodes = splitter.get_nodes_from_documents(docs)

    print(f"Total chunks: {len(nodes)}")

    # Extract text
    texts: List[str] = [node.get_content() for node in nodes]
    chunk_texts = texts

    print("Generating embeddings...")
    embeddings: EmbeddingArray = embedder.encode(
        texts, show_progress_bar=True
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Build FAISS index
    dimension: int = embeddings.shape[1]
    index: FaissIndex = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss_index = index

    print("Index built successfully!")
    return index, chunk_texts

def retrieve(query: str, top_k: int = 3) -> List[str]:
    global faiss_index, chunk_texts

    if faiss_index is None:
        build_index()

    assert faiss_index is not None  # for type checker

    query_embedding: EmbeddingArray = embedder.encode([query])
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    distances: np.ndarray
    indices: np.ndarray
    distances, indices = faiss_index.search(query_embedding, top_k)

    results: List[str] = []
    for i in indices[0]:
        if 0 <= i < len(chunk_texts):
            results.append(chunk_texts[i])

    return results

if __name__ == "__main__":
    build_index()

    print("\nTest retrieval:")
    results: List[str] = retrieve("caller feels numb and disconnected")

    for i, r in enumerate(results, start=1):
        print(f"\n--- Chunk {i} ---\n{r[:300]}")