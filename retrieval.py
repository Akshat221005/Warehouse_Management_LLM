print("retrieval.py is being executed")

import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve_with_rerank(query: str, top_k: int = 10, min_score: float = 0.3):
    index = faiss.read_index("vectorstore/index.faiss")

    with open("vectorstore/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    q_emb = EMBED_MODEL.encode([query])

    D, I = index.search(q_emb, top_k)

    distances = D[0]
    candidates = [chunks[i] for i in I[0]]

    # Pair chunks with distances
    pairs = list(zip(candidates, distances))

    # Filter out very bad matches
    filtered = [(c, d) for c, d in pairs if d < min_score]

    if len(filtered) == 0:
        return []

    # Rerank
    texts = [c for c, d in filtered]
    ce_pairs = [[query, t] for t in texts]
    scores = RERANKER.predict(ce_pairs)

    reranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)

    return [c for c, s in reranked[:5]]
