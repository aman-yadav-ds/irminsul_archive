from flask import Flask, render_template, request, jsonify
from src.pipeline.rag_engine import LoreReasoner # <-- Import your RAG class

app = Flask(__name__)

# Initialize the RAG engine globally
bot = LoreReasoner()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask_bot():
    data = request.json
    user_query = data.get('query', '').strip()

    if not user_query:
        return jsonify({"answer": "", "nodes": [], "edges": []})

    # 1. Ask the Hybrid RAG engine
    response = bot.ask(user_query)
    
    # 2. Extract Data
    text_answer = response.get('answer', "I couldn't process that.")
    graph_edges_data = response.get('graph_edges', [])

    # 3. Format for Vis.js
    nodes_set = set()
    edges = []

    for edge in graph_edges_data:
        source = edge['source']
        target = edge['target']
        relation = edge['relation']

        nodes_set.add(source)
        nodes_set.add(target)
        edges.append({
            "from": source,
            "to": target,
            "label": relation,
            "arrows": "to"
        })

    nodes = [{"id": name, "label": name} for name in nodes_set]

    return jsonify({
        "answer": text_answer,
        "nodes": nodes,
        "edges": edges
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)