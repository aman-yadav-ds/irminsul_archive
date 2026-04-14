import chromadb
from chromadb.utils import embedding_functions


class EntityResolver:
    def __init__(self, collection_name="genshin_entities"):
        # Local persistent storage
        self.client = chromadb.PersistentClient(path="./data/chroma_db")
        # Use a lightweight model for fast local string matching
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.emb_fn
        )

    def resolve_name(self, raw_name, threshold=0.85):
        """
        Takes a raw name and returns the canonical version from the DB if it exists.
        """
        results = self.collection.query(
            query_texts=[raw_name],
            n_results=1
        )

        # Check if we have a close match
        if (
            results.get('documents') 
            and len(results['documents']) > 0
            and len(results['documents'][0]) > 0
            and results.get('distances')
            and len(results['distances']) > 0
            and len(results['distances'][0]) > 0
        ):
            distance = results['distances'][0][0]
            if distance < (1 - threshold):
                return results['documents'][0][0]


        # If no match, add this new name as a canonical reference
        if raw_name not in self.collection.get(ids=[raw_name])['ids']:
            self.collection.add(
                documents=[raw_name],
                ids=[raw_name]
            )

        return raw_name