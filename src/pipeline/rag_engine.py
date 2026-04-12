import os
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class LoreReasoner:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )

        self.llm = ChatOllama(
            model="qwen2.5-coder:7b",
            temperature=0.3,
            
        )

        # CORRECTED TEMPLATE
        # Notice we use {{name: ...}} for examples so LangChain doesn't crash.
        self.cypher_generation_template = """
        Task: Generate Cypher statement to query a graph database.
        
        Schema:
        {schema}
        
        Instructions:
        1. **Entity Profiling (Who is / What is X)**: 
           - Example: MATCH (e:Entity)-[r]-(related:Entity) 
                      WHERE e.name =~ '(?i).*Scaramouche.*' 
                      RETURN startNode(r).name AS Subject, type(r) AS Action, endNode(r).name AS Object
        2. **Membership Discovery**: If the question asks "Who are..." or "List the members of..." a group:
           - First, find the group node.
           - Then, return all entities connected to it.
           - Example: MATCH (group:Entity)-[r]-(member:Entity) 
                      WHERE group.name =~ '(?i).*Eight.*Adepts.*' 
                      RETURN member.name, type(r)
        3. **Fuzzy Search**: ALWAYS use `CONTAINS` or regex `(?i).*Name.*` for names to catch slight variations. Avoid strict `=` matching unless absolutely necessary.
        4. **Wildcards**: Use `-[r]-` (undirected) to find all possible connections if the specific relationship type is unknown.
        
        Examples: 
        Question: Who is in the Adventurers' Guild?
        Cypher: MATCH (m:Entity)-[r]-(g:Entity) WHERE g.name =~ '(?i).*Adventurer.*Guild.*' RETURN m.name, type(r)

        Question: What are the Adepti?
        Cypher: MATCH (e:Entity)-[r]-(related:Entity) WHERE e.name =~ '(?i).*Adepti.*' RETURN e.name, e.aliases, type(r), related.name
        
        CRITICAL:
        - Return ONLY the valid Cypher query.
        - Do not add explanations or markdown blocks around the query.
        - Combine results into one return statement.
        
        Question: {question}
        Cypher Query:
        """

        self.cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"], 
            template=self.cypher_generation_template
        )

        self.chain = GraphCypherQAChain.from_llm(
            self.llm,
            graph=self.graph,
            cypher_prompt=self.cypher_prompt,
            verbose=True,
            allow_dangerous_requests=True,
            return_direct=False,
            top_k=50
        )

    def get_dynamic_schema(self):
        # This query fetches all unique relationship types actually in your DB
        result = self.graph.query("CALL db.relationshipTypes()")
        rel_types = [row['relationshipType'] for row in result]
        
        # This fetches all property keys (like 'name', 'aliases', 'label')
        props = self.graph.query("CALL db.propertyKeys()")
        prop_keys = [row['propertyKey'] for row in props]

        return f"Existing Relationships: {rel_types}\nAvailable Properties: {prop_keys}"

    def ask(self, question):
        try:
            print(f"🤔 Thinking: {question}")
            # .invoke returns a dict, we want the 'result' key
            response = self.chain.invoke({"query": question})
            return response['result']
        except Exception as e:
            return f"I tripped over a vine (Graph Error): {e}"

if __name__ == "__main__":
    bot = LoreReasoner()
    
    # 2. Test "Connection" logic
    print(bot.ask("Why is the Abyss so important in the lore?"))