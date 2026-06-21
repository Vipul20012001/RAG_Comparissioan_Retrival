import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
class SimpleVectorStore:
    def __init__(self, embeddings=None, metadatas=None):
        self.embeddings = np.array(embeddings) if embeddings is not None else np.zeros((0, 0), dtype=np.float32)
        self.metadatas = metadatas or []

    def add(self, texts, metadatas, model):
        new_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.metadatas.extend(metadatas)

    def search(self, query, model, top_k=4):
        if self.embeddings.size == 0:
            return []
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        indexes = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "score": float(scores[i]),
                "text": self.metadatas[i]["text"],
                "source": self.metadatas[i]["source"],
            }
            for i in indexes
        ]

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "metadatas": self.metadatas}, f)

    @classmethod
    def load(cls, path):
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(embeddings=data.get("embeddings"), metadatas=data.get("metadatas"))
