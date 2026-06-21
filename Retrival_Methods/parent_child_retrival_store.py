import json
import os
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MultiVectorStore:
    def __init__(self):
        """Initializes storage structures for decoupled parent-child relationships."""
        self.embeddings = np.empty((0, 0)) # Matrix storing CHILD vector representations
        self.child_to_parent_map = []      # Maps child index to parent_id: [{"parent_id": "...", "text": "child text"}]
        self.parent_store = {}             # Main dictionary: {parent_id: "Large full context block"}

    def add_documents(self, parent_text, child_texts, child_embeddings, source_metadata="Unknown"):
        """
        Registers a large parent text block along with its precomputed smaller child embeddings.
        
        Args:
            parent_text (str): The full context paragraph/page to feed to the LLM.
            child_texts (list of str): Small subdivisions or summaries extracted from parent.
            child_embeddings (list or np.ndarray): Precomputed vectors for each child text.
            source_metadata (str): Optional tracking identifier (e.g., filename).
        """
        # 1. Generate a unique identity token for this parent block
        parent_id = str(uuid.uuid4())
        self.parent_store[parent_id] = {
            "text": parent_text,
            "source": source_metadata
        }

        # 2. Append child embeddings to vector matrix
        new_child_embeddings = np.array(child_embeddings)
        if self.embeddings.size == 0:
            self.embeddings = new_child_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_child_embeddings])
        
        # 3. Map each new child slot back to its parent container id
        for text in child_texts:
            self.child_to_parent_map.append({
                "parent_id": parent_id,
                "child_text": text
            })

    def search_parent_chunks(self, query, model, top_k=2):
        """
        Searches small precise children, but retrieves and returns unique large parent chunks.
        
        Args:
            query (str): The incoming search string.
            model: Embedding model instance to encode query text.
            top_k (int): Target count of unique parent contextual blocks to deliver.
            
        Returns:
            list: List of larger parent text string chunks containing matching context.
        """
        if self.embeddings.size == 0 or not self.child_to_parent_map:
            return []

        # Step 1: Search against child embeddings
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        sorted_child_indexes = np.argsort(scores)[::-1]

        # Step 2: Extract parent IDs while enforcing uniqueness
        retrieved_parent_chunks = []
        seen_parents = set()

        for idx in sorted_child_indexes:
            # Check if this child score is valid
            if scores[idx] <= 0:
                continue
                
            associated_parent_id = self.child_to_parent_map[idx]["parent_id"]
            
            # Prevent pulling the exact same parent chunk multiple times
            if associated_parent_id not in seen_parents:
                seen_parents.add(associated_parent_id)
                parent_data = self.parent_store[associated_parent_id]
                retrieved_parent_chunks.append(parent_data["text"])
            
            # Break early once target parent count limit is satisfied
            if len(retrieved_parent_chunks) >= top_k:
                break

        return retrieved_parent_chunks

    def save(self, filename_base):
        """Saves both the dense index and mapping databases into a single JSON file."""
        filepath = str(filename_base) + ".json"
        data = {
            "embeddings": self.embeddings.tolist(),
            "child_to_parent_map": self.child_to_parent_map,
            "parent_store": self.parent_store
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, filename_base):
        """Reconstructs the multi-vector matrix configurations from disk safely."""
        filepath = str(filename_base) + ".json"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Multi-Vector database missing at path location: {filepath}")
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        instance = cls()
        instance.embeddings = np.array(data["embeddings"])
        instance.child_to_parent_map = data["child_to_parent_map"]
        instance.parent_store = data["parent_store"]
        return instance
