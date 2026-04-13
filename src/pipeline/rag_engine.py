import os
import chromadb
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

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

        # 3. LLM Setup
        self.llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

        # --- THE GRAPH PROMPT ---
        # --- THE RESTRICTIVE GRAPH PROMPT ---
        self.cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"], 
            template="""
            Task: Generate a Cypher statement to query a graph database.
            
            CRITICAL RULES:
            1. You MUST use the Full-Text search index called `entitySearch`.
            2. Append a `~` to the end of the search keyword to enable fuzzy matching (e.g., `Traveller~`).
            3. Your query MUST follow this exact structure:
            
            CALL db.index.fulltext.queryNodes("entitySearch", "<INSERT_KEYWORD_HERE>~") YIELD node AS n
            MATCH (n)-[r]-(m:Entity)
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
            input_variables=["context", "question"],
            template="""
            You are a Genshin Impact lore archivist. 
            Answer the user's question based ONLY on the provided text context below. 
            If the context does not contain the answer, explicitly state: "I don't have enough information in my archives to answer that."
            Do not invent lore.

            CONTEXT (Raw Source Texts):
            {context}

            USER QUESTION:
            {question}

            FINAL ANSWER:
            """
        )

    def ask(self, question):
        print(f"🤔 Step 1: Querying Graph for structural context: {question}")
        
        # 1. Get Graph Data
        graph_response = {}
        try:
            graph_response = self.graph_chain.invoke({"query": question})
        except Exception as e:
            print(f"Graph Error: {e}")

        # 2. Extract Graph Edges securely (The Bulletproof Parser)
        raw_db_results = []
        for step in graph_response.get('intermediate_steps', []):
            if isinstance(step, list): raw_db_results.extend(step)
            elif isinstance(step, dict) and 'context' in step: raw_db_results.extend(step['context'])

        search_terms = []
        graph_edges = []
        
        for record in raw_db_results:
            if isinstance(record, dict):
                values = list(record.values())
                if len(values) >= 3:
                    source, relation, target = str(values[0]), str(values[1]), str(values[2])
                    # Add to graph UI payload
                    graph_edges.append({"source": source, "relation": relation, "target": target})
                    # Build search string for Vector DB
                    search_terms.append(f"{source} {relation.replace('_', ' ')} {target}")

        # 3. Vector Search Strategy
        # If the graph found connections, search for those exact connections. 
        # If the graph failed, fall back to searching the user's original question.
        vector_query = " ".join(search_terms) if search_terms else question
        print(f"🔎 Step 2: Querying Vector DB with: {vector_query[:100]}...")

        # Fetch the top 3 most relevant raw text chunks
        vector_results = self.vector_collection.query(
            query_texts=[vector_query],
            n_results=3
        )

        # 4. Generate Final Answer
        # Join the retrieved paragraphs into one big string
        retrieved_text = ""
        if vector_results['documents'] and vector_results['documents'][0]:
            retrieved_text = "\n\n---\n\n".join(vector_results['documents'][0])

        print(f"✍️ Step 3: Generating final grounded answer...")
        final_chain = self.final_answer_prompt | self.llm
        final_llm_response = final_chain.invoke({
            "context": retrieved_text,
            "question": question
        })

        return {
            "answer": final_llm_response.content, # The grounded text
            "graph_edges": graph_edges            # The UI visual data
        }


if __name__ == "__main__":
    bot = LoreReasoner()
    
    # 2. Test "Connection" logic
    print(bot.ask("How is Skirk and Surtologi related?"))