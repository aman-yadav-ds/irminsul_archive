import os
import glob
import chromadb
from chromadb.utils import embedding_functions

class VectorDBBuilder:
    def __init__(self, raw_data_dir="data/raw", db_path="./data/chroma_db"):
        self.raw_data_dir = raw_data_dir
        
        print("🔌 Connecting to local ChromaDB...")
        # This creates (or connects to) the folder on your hard drive
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # We use Chroma's default embedding function (SentenceTransformers: all-MiniLM-L6-v2)
        # It is tiny, incredibly fast, runs locally, and requires zero API keys.
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create the collection we referenced in the LoreReasoner
        self.collection = self.chroma_client.get_or_create_collection(
            name="raw_lore",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def chunk_text_with_overlap(self, text, chunk_size=1000, overlap=200):
        """Splits text into chunks, preserving some context between them."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunks.append(text[start:end])
            # Move the start forward, but step back by the overlap amount
            start += chunk_size - overlap
            
        return chunks

    def ingest_files(self):
        files = glob.glob(os.path.join(self.raw_data_dir, "*.txt"))
        if not files:
            print(f"⚠️ No .txt files found in {self.raw_data_dir}")
            return

        print(f"📂 Found {len(files)} files. Starting ingestion...")

        total_chunks_added = 0

        for filepath in files:
            filename = os.path.basename(filepath)
            print(f"📖 Processing {filename}...")
            
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if len(content.strip()) < 50:
                continue

            # 1. Chunk the document
            chunks = self.chunk_text_with_overlap(content)
            
            # 2. Prepare the data for ChromaDB
            ids = []
            documents = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                # Unique ID for every single chunk
                chunk_id = f"{filename}_chunk_{i}"
                ids.append(chunk_id)
                documents.append(chunk)
                # Save the source file name so we know exactly where the lore came from
                metadatas.append({"source": filename, "chunk_index": i})

            # 3. Batch Upsert to Vector Database
            # Upsert will add them if they are new, or update them if the IDs already exist
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            total_chunks_added += len(chunks)
            print(f"   ✅ Embedded and saved {len(chunks)} chunks.")

        print(f"\n🎉 Vector Database build complete! Total chunks embedded: {total_chunks_added}")
        print("Your Hybrid RAG engine is now ready to query.")

if __name__ == "__main__":
    builder = VectorDBBuilder()
    builder.ingest_files()