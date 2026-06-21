import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

class SimpleVectorStore:
    def __init__(self):
        """Initializes an empty vector store."""
        self.embeddings = np.empty((0, 0)) # Matrix storing numerical representations
        self.metadatas = []                # List of dicts: [{"text": "...", "source": "..."}]

    def add_documents(self, embeddings_list, metadatas_list):
        """
        Adds text chunks and their matching embeddings to the store.
        
        Args:
            embeddings_list (list or np.ndarray): Calculated vector array.
            metadatas_list (list): Metadata structures containing 'text' and 'source'.
        """
        new_embeddings = np.array(embeddings_list)
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.metadatas.extend(metadatas_list)

    def search_hybrid(self, query, model, top_k=4, alpha=0.5, rrf_k=60):
        """
        Executes a unified hybrid retrieval matching process.
        
        Args:
            query (str): The search phrase.
            model: The embedding model instance (e.g., SentenceTransformer).
            top_k (int): Maximum number of context chunks to pull.
            alpha (float): Search weight scale. 1.0 is pure vector, 0.0 is pure keyword.
            rrf_k (int): RRF constant parameter to mitigate outlier rank impacts.
            
        Returns:
            list: List of retrieved text string chunks sorted by combined relevance.
        """
        if self.embeddings.size == 0 or not self.metadatas:
            return []

        # ---------------------------------------------------------
        # STEP 1: Dense Vector Search (Semantic)
        # ---------------------------------------------------------
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        dense_scores = cosine_similarity(query_embedding, self.embeddings)[0]
        dense_ranked_indexes = np.argsort(dense_scores)[::-1]

        # ---------------------------------------------------------
        # STEP 2: Sparse Keyword Search (BM25 Matcher)
        # ---------------------------------------------------------
        corpus = [meta["text"] for meta in self.metadatas]
        tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = query.lower().split(" ")
        sparse_scores = bm25.get_scores(tokenized_query)
        sparse_ranked_indexes = np.argsort(sparse_scores)[::-1]

        # ---------------------------------------------------------
        # STEP 3: Reciprocal Rank Fusion Merger
        # ---------------------------------------------------------
        rrf_scores = {i: 0.0 for i in range(len(self.metadatas))}

        for rank, idx in enumerate(dense_ranked_indexes):
            rrf_scores[idx] += alpha * (1.0 / (rrf_k + rank + 1))

        for rank, idx in enumerate(sparse_ranked_indexes):
            rrf_scores[idx] += (1 - alpha) * (1.0 / (rrf_k + rank + 1))

        # Sort entries based on total fusion performance
        sorted_indexes = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        # ---------------------------------------------------------
        # STEP 4: Return List of the Text Chunks
        # ---------------------------------------------------------
        return [
            self.metadatas[i]["text"]
            for i in sorted_indexes 
            if (dense_scores[i] > 0 or sparse_scores[i] > 0)
        ]

    def save(self, filename_base):
        """Saves embeddings matrix and document text structures into a storage file."""
        filepath = str(filename_base) + ".json"
        data = {
            "embeddings": self.embeddings.tolist(),
            "metadatas": self.metadatas
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, filename_base):
        """Loads a persistent store index from file, explicitly fixing Path string casting."""
        filepath = str(filename_base) + ".json"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No existing index index database found at: {filepath}")
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        instance = cls()
        instance.embeddings = np.array(data["embeddings"])
        instance.metadatas = data["metadatas"]
        return instance