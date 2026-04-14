import os
import json
import time
import glob
import re
import chromadb
from langchain_ollama import ChatOllama
from src.utils.neo4j_client import Neo4jClient
from src.utils.entity_resolver import EntityResolver

class LoreExtractor:
    def __init__(self):
        self.db = Neo4jClient()
        self.db.connect()

        self.entity_resolver = EntityResolver()

        self.model_name = "qwen2.5-coder:7b"
        
        self.llm = ChatOllama(
            model=self.model_name, 
            temperature=0,
            format="json" 
        )
        
        # --- Initialize ChromaDB ---
        self.chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="genshin_aliases",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.known_entities = self._load_known_entities_from_chroma()
        
        # --- NEW: Tracker for Resumability ---
        self.processed_log_path = "data/processed_files.log"

    def _load_known_entities_from_chroma(self):
        print("🔍 Loading existing aliases from ChromaDB...")
        existing_data = self.collection.get(include=['metadatas'])
        known_entities = {}
        
        if existing_data and existing_data['metadatas']:
            for meta in existing_data['metadatas']:
                canonical_name = meta['canonical_name']
                aliases = json.loads(meta['aliases_json'])
                known_entities[canonical_name] = aliases
                
        print(f"📦 Loaded {len(known_entities)} entities from vector storage.")
        return known_entities

    def chunk_text(self, text, chunk_size=3000):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def clean_json_string(self, json_str):
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        return json_str

    # --- NEW: Helper method to mark files as done ---
    def _mark_file_completed(self, filename):
        with open(self.processed_log_path, "a", encoding="utf-8") as f:
            f.write(filename + "\n")

    def process_directory(self, dir_path="data/raw"):
        files = glob.glob(os.path.join(dir_path, "*.txt"))
        
        # --- NEW: Read the tracker log before starting ---
        processed_files = set()
        if os.path.exists(self.processed_log_path):
            with open(self.processed_log_path, "r", encoding="utf-8") as f:
                processed_files = set(line.strip() for line in f)

        print(f"📂 Found {len(files)} total files.")
        print(f"⏩ Found {len(processed_files)} previously completed files. Skipping those.")
        print(f"🚀 Starting Local Extraction {self.model_name}...")

        # FIXED: Removed 'enumerate' so filepath receives the string properly
        for filepath in files: 
            filename = os.path.basename(filepath)
            
            # --- NEW: The Resume Check ---
            if filename in processed_files:
                continue # Silently skip to the next file
            
            print(f"\n📖 Reading {filename}...")
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                if len(content) < 100: 
                    self._mark_file_completed(filename) # Mark empty files as done so we don't retry them
                    continue

                chunks = self.chunk_text(content)
                print(f"   🧩 Split into {len(chunks)} chunks.")

                for i, chunk in enumerate(chunks):
                    print(f"   🤖 Processing chunk {i+1}/{len(chunks)}...")
                    self.extract_and_upload(chunk, chunk_index=i, source_file=filename)

                # --- NEW: Success! Mark this file as completely finished ---
                # We only reach this line if NO exceptions were thrown during chunk processing
                self._mark_file_completed(filename)
                print(f"   🏁 Finished entirely with {filename}. Saved to checkpoint log.")

            except Exception as e:
                # If an error happens, we do NOT mark it as completed.
                # This ensures the script will try it again the next time you run it.
                print(f"   ❌ Error processing {filename}: {e}")

    def extract_and_upload(self, text, chunk_index=0, source_file="Unknown"):
        # Retrieve ONLY relevant aliases from ChromaDB
        relevant_context = {}
        
        if self.collection.count() > 0:
            results = self.collection.query(
                query_texts=[text],
                n_results=min(15, self.collection.count()) 
            )
            
            if results['metadatas'] and results['metadatas'][0]:
                for meta in results['metadatas'][0]:
                    aliases = json.loads(meta['aliases_json'])
                    if aliases: 
                        relevant_context[meta['canonical_name']] = aliases

        known_aliases_str = json.dumps(relevant_context, indent=2, ensure_ascii=False) if relevant_context else "{}"

        prompt = f"""
        You are a precise linguistics and lore extraction AI. Extract Genshin Impact lore entities and relationships from the text below.
        
        CRITICAL: Output MUST be valid JSON.
        
        === RELEVANT KNOWN ENTITIES AND ALIASES ===
        Below is a dynamically fetched dictionary of canonical names and their known aliases relevant to this text. 
        If you encounter an alias in the text, you MUST use its canonical name for the "canonical_name" field.
        And you MUST use the known aliases to help resolve any ambiguous references in the text.
        {known_aliases_str}
        ===========================================

        Schema Example:
        {{
            "reasoning": "First, briefly write out who is performing the action and who is receiving it to establish the correct direction.",
            "entities": [
                {{"canonical_name": "Name", "aliases": ["Alias1"], "label": "Type"}}
            ],
            "relationships": [
                {{"source": "Name1", "target": "Name2", "type": "RELATIONSHIP_TYPE"}}
            ]
        }}

        STRICT EXTRACTION RULES:
        1. PROPER NOUNS ONLY: You MUST ONLY extract specific, named entities. DO NOT extract generic nouns.
        2. EXPLICIT FACTS ONLY: Only extract relationships explicitly stated in the text.
        3. STRICT DIRECTIONALITY: The "source" performs the action, and the "target" receives it. 
        4. REVERSING PASSIVE VOICE: If the text uses passive voice, you MUST reverse the direction so the true Source performs the action.
           - Text: "Liyue was created by Morax." 
           - CORRECT: Source="Morax", Target="Liyue", Type="CREATED"
           - INCORRECT: Source="Liyue", Target="Morax", Type="CREATED_BY"
        5. ACTIVE VOICE: Use specific, active-voice verbs in ALL CAPS for relationships.

        Now, process the following Text carefully and apply the rules:
        {text}
        """

        try:
            response = self.llm.invoke(prompt)
            clean_content = self.clean_json_string(response.content)
            data = json.loads(clean_content)
            
            if "reasoning" in data:
                print(f"      🧠 Logic: {data['reasoning']}")

            os.makedirs("data/processed", exist_ok=True)
            json.dump(data, open(f"data/processed/{source_file.replace('.txt', '')}_{chunk_index}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

            count_ent = 0
            entities_to_upsert = set() 

            # Process & Upload Entities
            for entity in data.get('entities', []):
                original_name = entity.get('canonical_name', "")

                if not original_name or original_name.islower():
                    print(f"      🗑️ Dropped generic/invalid entity: '{original_name}'")
                    continue

                extracted_aliases = entity.get('aliases', [])
                resolved_name = self.entity_resolver.resolve_name(original_name)

                if resolved_name != original_name:
                    print(f"      🔍 Resolved '{original_name}' to '{resolved_name}'")
                
                is_new_or_updated = False
                if resolved_name not in self.known_entities:
                    self.known_entities[resolved_name] = []
                    is_new_or_updated = True
                
                for alias in extracted_aliases:
                    if alias not in self.known_entities[resolved_name] and alias != resolved_name:
                        self.known_entities[resolved_name].append(alias)
                        is_new_or_updated = True
                
                if is_new_or_updated:
                    entities_to_upsert.add(resolved_name)
                
                entity['canonical_name'] = resolved_name
                
                entity_cypher = """
                MERGE (e:Entity {name: $canonical_name})
                ON CREATE SET
                    e.aliases = $aliases,
                    e.label = $label,
                    e.source_file = $source
                ON MATCH SET
                    e.aliases = apoc.coll.toSet(coalesce(e.aliases, []) + coalesce($aliases, []))
                """
                entity['source'] = source_file
                self.db.query(entity_cypher, parameters=entity)
                count_ent += 1

            # Batch Upsert newly discovered/updated aliases to ChromaDB
            if entities_to_upsert:
                ids = []
                documents = []
                metadatas = []
                
                for name in entities_to_upsert:
                    ids.append(name)
                    searchable_text = f"{name} " + " ".join(self.known_entities[name])
                    documents.append(searchable_text)
                    metadatas.append({
                        "canonical_name": name,
                        "aliases_json": json.dumps(self.known_entities[name])
                    })
                
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )

            # Process & Upload Relationships
            count_rel = 0
            for rel in data.get('relationships', []):
                if not rel.get('source') or not rel.get('target'): continue
                if not re.match(r'^[A-Z_]+$', rel['type']): continue
                
                rel['source'] = self.entity_resolver.resolve_name(rel['source'])
                rel['target'] = self.entity_resolver.resolve_name(rel['target'])

                rel_cypher = f"""
                MERGE (a:Entity {{name: $source}})
                MERGE (b:Entity {{name: $target}})
                MERGE (a)-[:{rel['type']}]->(b)
                """
                self.db.query(rel_cypher, parameters=rel)
                count_rel += 1
            
            print(f"      ✅ Extracted {count_ent} entities, {count_rel} relations.")

        except json.JSONDecodeError:
            print(f"      ⚠️ Model failed to generate valid JSON. Skipping chunk.")
        except Exception as e:
            print(f"      ⚠️ Error during extraction/upload: {e}")

if __name__ == "__main__":
    extractor = LoreExtractor()
    extractor.process_directory()