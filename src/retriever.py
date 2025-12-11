# src/retriever.py

from sentence_transformers import util, SentenceTransformer
import numpy as np
import faiss

def build_faiss_index(df):
    """
    Build FAISS index from df['vector_embedding'].
    """
    vectors = np.array(df["vector_embedding"].tolist()).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve_top_sections(query, embedder, df, index, k=3):
    """
    Returns list of top-k most relevant OSH sections.
    """
    qv = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(qv, k)

    results = []
    for idx in indices[0]:
        results.append(df.iloc[idx]["text"])

    return results
