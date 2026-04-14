import os
import chromadb
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import json

load_dotenv()

class LoreReasoner:
    def __init__(self):
        # 1. Graph DB Setup
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )

        # 2. Vector DB Setup (Assuming you embed raw texts into a 'raw_lore' collection)
        self.chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        self.vector_collection = self.chroma_client.get_or_create_collection("raw_lore")

        # 3. LLM Setup: The "Split-Brain" Architecture

        # --- THE EXTRACTOR (Left Brain) ---
        # Highly restricted, robotic, and deterministic. 
        # Goal: Never hallucinate, only extract exact strings.
        self.graph_query_llm = ChatOllama(
            model="qwen2.5-coder:7b", 
            temperature=0.0,
            top_p=0.1,          
            top_k=10,           
            num_predict=50,     # Keep it short, it only needs to output a JSON list
            format="json"
        )

        # --- THE STORYTELLER (Right Brain) ---
        # Creative, fluent, and narrative-driven. 
        # Goal: Weave facts from the vector/graph DB into a beautiful lore explanation.
        self.final_answer_llm = ChatOllama(
            model="qwen2.5-coder:7b", # You can even swap this to a different model later!
            temperature=0.4,          # Enough heat to be creative, not enough to hallucinate facts
            top_p=0.9,                # Let it use a wider vocabulary for better storytelling
            top_k=40
        )

        # 4. Alias Collection for Entity Resolution
        self.aliases_collection = self.chroma_client.get_or_create_collection(
            name="genshin_aliases",
            metadata={"hnsw:space": "cosine"}
        )

        # --- THE GRAPH PROMPT ---
        # --- THE RESTRICTIVE GRAPH PROMPT ---
        self.cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"], 
            template="""
            Task: Generate a standard Cypher statement to query a graph database.
            
            CRITICAL RULES:
            1. Use the EXACT canonical names provided in the "Hint" section of the question.
            2. Match against the `name` property of the `Entity` node (e.g., `n.name = 'Traveler'`).
            3. DO NOT hallucinate invalid syntax. Always use `type(r)` to get the relationship name in the RETURN clause.
            4. Your generated query MUST follow this exact structure:
            
            MATCH (n:Entity)-[r]-(m:Entity)
            WHERE n.name = 'INSERT_NAME_FROM_HINT_HERE'
            RETURN n.name AS source, type(r) AS relation, m.name AS target LIMIT 50
            
            Schema: {schema}
            Question: {question}
            Cypher Query:
            """
        )

        self.graph_chain = GraphCypherQAChain.from_llm(
            self.llm,
            graph=self.graph,
            cypher_prompt=self.cypher_prompt,
            verbose=True,
            allow_dangerous_requests=True,
            return_direct=False,
            return_intermediate_steps=True, # Keep raw DB results
            top_k=20
        )

        # --- NEW: THE VECTOR RAG PROMPT ---
        self.final_answer_prompt = PromptTemplate(
            input_variables=["text_context", "graph_context", "question"],
            template="""
            You are a Genshin Impact lore archivist. 
            Answer the user's question using the provided Text Archives and Graph Database Connections.
            
            If the context does not contain the answer, explicitly state: "I don't have enough information in my archives to answer that."
            Do not invent lore.

            === GRAPH DATABASE CONNECTIONS (Structured Facts) ===
            {graph_context}

            === TEXT ARCHIVES (Lore Paragraphs) ===
            {text_context}

            USER QUESTION:
            {question}

            FINAL ANSWER:
            """
        )

    def ask(self, question):
        # 1. Get the exact names using your Vector dictionary
        original_question, canonical_names = self.fix_spelling_mistakes(question)
        
        graph_edges = []
        search_terms = []
        
        print(f"🤔 Step 1: Querying Graph directly for {canonical_names}...")
        
        # 2. Hardcoded, bulletproof Graph Query (No LLM required!)
        for name in canonical_names:
            # We use a parameterized query ($name) for safety and speed
            cypher_query = """
            MATCH (n:Entity {name: $name})-[r]-(m:Entity)
            RETURN n.name AS source, type(r) AS relation, m.name AS target LIMIT 20
            """
            
            try:
                # self.graph.query is a built-in LangChain Neo4j method
                results = self.graph.query(cypher_query, params={"name": name})
                
                for record in results:
                    source = record['source']
                    relation = record['relation']
                    target = record['target']
                    
                    graph_edges.append({"source": source, "relation": relation, "target": target})
                    search_terms.append(f"{source} {relation.replace('_', ' ')} {target}")
            
            except Exception as e:
                print(f"Graph Error for {name}: {e}")

        # 3. Vector Search Strategy
        print(f"🔎 Step 2: Querying Vector DB for lore text...")
        vector_results = self.vector_collection.query(
            query_texts=[original_question], # <-- Use the pure question here!
            n_results=3
        )

        retrieved_text = ""
        if vector_results['documents'] and vector_results['documents'][0]:
            retrieved_text = "\n\n---\n\n".join(vector_results['documents'][0])

        # --- FIX 2: Format the Graph Edges so the LLM can read them ---
        graph_context_strings = []
        for edge in graph_edges:
            graph_context_strings.append(f"{edge['source']} -[{edge['relation']}]-> {edge['target']}")
        
        formatted_graph_context = "\n".join(graph_context_strings) if graph_context_strings else "No direct connections found."

        print(f"✍️ Step 3: Generating final grounded answer...")
        
        # USE THE STORYTELLER LLM
        final_chain = self.final_answer_prompt | self.final_answer_llm
        
        final_llm_response = final_chain.invoke({
            "text_context": retrieved_text,
            "graph_context": formatted_graph_context, 
            "question": original_question
        })

        return {
            "answer": final_llm_response.content,
            "graph_edges": graph_edges
        }

    def fix_spelling_mistakes(self, question):
        print("🪄 Extracting entities and checking spelling...")
        
        prompt = f"""
        Extract the core Genshin Impact entities (characters, places, items) from the user's question.
        
        CRITICAL RULES:
        1. Extract the EXACT text the user typed.
        2. DO NOT correct the user's spelling.
        3. DO NOT guess, translate, or substitute names. 
        4. Ignore generic filler words.
        
        Question: "{question}"
        
        Output ONLY a valid JSON list of strings.
        """
        
        # USE THE EXTRACTOR LLM
        response = self.graph_query_llm.invoke(prompt)
        
        try:
            messy_names = json.loads(response.content)
        except:
            return question, [] # Fallback
            
        canonical_names_found = []
        
        for messy_name in messy_names:
            results = self.aliases_collection.query(
                query_texts=[messy_name],
                n_results=1 
            )
            
            if results['metadatas'] and results['metadatas'][0]:
                canonical_name = results['metadatas'][0][0]['canonical_name']
                if canonical_name not in canonical_names_found:
                    canonical_names_found.append(canonical_name)
                
        print(f"   ✅ Identified Canonical Entities: {canonical_names_found}")
        return question, canonical_names_found
if __name__ == "__main__":
    bot = LoreReasoner()
    
    # 2. Test "Connection" logic
    print(bot.ask("How is Skirk and Surtologi related?"))