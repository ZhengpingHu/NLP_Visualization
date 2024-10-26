import joblib
import numpy as np
from collections import defaultdict
from pyvis.network import Network

model_path = "../models/sentiment_cooccurrence_graph.pkl"
G = joblib.load(model_path)
print("Model loaded successfully.")

def infer_related_words(graph, keywords, sentiment="positive", top_n=10):
    sentiment_index = {"positive": 2, "neutral": 1, "negative": 0}
    selected_sentiment_idx = sentiment_index.get(sentiment, 2)

    related_words = defaultdict(float)

    for keyword in keywords:
        if keyword in graph:
            neighbors = graph.neighbors(keyword)
            for neighbor in neighbors:
                edge_data = graph.get_edge_data(keyword, neighbor)
                if edge_data:
                    related_words[neighbor] += edge_data['weight']

    sorted_related_words = sorted(related_words.items(), key=lambda item: item[1], reverse=True)
    return [word for word, _ in sorted_related_words[:top_n]]

def visualize_graph_3d(graph, target_keywords=None, output_file="interactive_graph.html"):
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white", heading="Word Co-occurrence Graph")

    for node in graph.nodes(data=True):
        if target_keywords is None or node[0] in target_keywords:
            net.add_node(node[0], title=str(node[1].get("features")), color="#97C2FC")

    for edge in graph.edges(data=True):
        net.add_edge(edge[0], edge[1], title=str(edge[2]["weight"]), color="gray")

    net.show_buttons(filter_=["physics"])
    net.show(output_file)
    print(f"Graph visualization saved to {output_file}")


while True:
    input_keywords = input("Enter keywords (comma-separated): ").split(",")
    input_keywords = [kw.strip() for kw in input_keywords]
    input_sentiment = input("Enter sentiment (positive, neutral, negative): ").strip().lower()

    related_words = infer_related_words(G, input_keywords, input_sentiment, top_n=10)
    print(f"Related words for {input_keywords} with {input_sentiment} sentiment are: {related_words}")



    visualize_graph_3d(G, target_keywords=input_keywords)
    cont = input("Do you want to enter another set of keywords? (yes/no): ").strip().lower()
    if cont != "yes":
        break
