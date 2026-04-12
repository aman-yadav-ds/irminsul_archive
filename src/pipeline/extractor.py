import os
import json
import time
import glob
import re
from langchain_ollama import ChatOllama # CHANGED: Switched from Google to Ollama
from src.utils.neo4j_client import Neo4jClient
from src.utils.entity_resolver import EntityResolver

class LoreExtractor:
    def __init__(self):
        self.db = Neo4jClient()
        self.db.connect()

        self.entity_resolver = EntityResolver()

        self.model_name = "qwen2.5-coder:7b"
        
        # CHANGED: Initialize Local LLM
        # "format": "json" is CRITICAL. It forces the model to only output valid JSON.
        # temperature=0 makes it deterministic (less creative, more precise).
        self.llm = ChatOllama(
            model=self.model_name, 
            temperature=0,
            format="json" 
        )


    def chunk_text(self, text, chunk_size=6000):
        # CHANGED: Local models have smaller context windows than Gemini.
        # Reduced chunk_size to 6000 chars to be safe for 8k context window.
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def clean_json_string(self, json_str):
        """Helper to strip markdown if the model adds it despite instructions."""
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        return json_str

    def process_directory(self, dir_path="data/raw"):
        files = glob.glob(os.path.join(dir_path, "*.txt"))
        print(f"📂 Found {len(files)} files. Starting Local Extraction {self.model_name}...")

        for i, filepath in enumerate(files):
            if i == 1: break # For testing, only process the first 1 files. Remove this line for full run.
            filename = os.path.basename(filepath)
            print(f"\n📖 Reading {filename}...")
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                if len(content) < 100: continue

                chunks = self.chunk_text(content)
                print(f"   🧩 Split into {len(chunks)} chunks.")

                for i, chunk in enumerate(chunks):
                    print(f"   🤖 Processing chunk {i+1}/{len(chunks)}...")
                    self.extract_and_upload(chunk, chunk_index=i, source_file=filename)
                    # No time.sleep() needed! You own the hardware.

            except Exception as e:
                print(f"   ❌ Error processing {filename}: {e}")

    def extract_and_upload(self, text, chunk_index=0, source_file="Unknown"):
        # SIMPLIFIED PROMPT: Smaller models need less "fluff" and more concrete examples.
        prompt = f"""
        You are a precise linguistics and lore extraction AI. Extract Genshin Impact lore entities and relationships from the text below.
        
        CRITICAL: Output MUST be valid JSON.
        
        Schema:
        {{
            "entities": [
                {{"canonical_name": "Name", "aliases": ["Alias1"], "label": "Type"}}
            ],
            "relationships": [
                {{"source": "Name1", "target": "Name2", "type": "RELATIONSHIP_TYPE"}}
            ]
        }}

        STRICT RULES:
        1. ONLY extract relationships explicitly stated as a fact in the Text. Do not invent or deduce relationships.
        2. You MUST USE active voice to describe relationships. For example, use "CREATOR_OF" instead of "DESCENDED_FROM". This ensures clarity and consistency in the graph.
        3. Use ALL CAPS for relationship types and avoid vague terms like "RELATED_TO". Be as specific as possible (e.g., "CREATED", "OPPOSED_TO", "ALLIED_WITH").
        4. Ensure the relationships point to target entities that are actually mentioned in the text.

        EXAMPLE:
        Text: "The Abyss holds the power of the Void Realm, which is diametrically opposed to the powers of the Light Realm."
        Output:
        {{
            "entities": [
                {{"canonical_name": "Abyss", "aliases": ["Void Realm"], "label": "Realm"}},
                {{"canonical_name": "Light Realm", "aliases": [], "label": "Realm"}}
            ],
            "relationships": [
                {{"source": "Abyss", "target": "Light Realm", "type": "OPPOSED_TO"}}
            ]
        }}

        Now, process the following Text:
        {text}
        """

        try:
            response = self.llm.invoke(prompt)
            clean_content = self.clean_json_string(response.content)
            data = json.loads(clean_content)

            os.makedirs("data/processed", exist_ok=True)
            json.dump(data, open(f"data/processed/{source_file.replace('.txt', '')}_{chunk_index}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

            # 1. Upload Entities
            count_ent = 0
            for entity in data.get('entities', []):
                original_name = entity.get('canonical_name', "")

                resolved_name = self.entity_resolver.resolve_name(original_name)

                if resolved_name != original_name:
                    print(f"      🔍 Resolved '{original_name}' to '{resolved_name}'")
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

            # 2. Upload Relationships
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
            print(f"      ⚠️ Neo4j Error: {e}")

if __name__ == "__main__":
    extractor = LoreExtractor()
    extractor.process_directory()