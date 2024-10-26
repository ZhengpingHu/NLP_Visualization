import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import re
import spacy
from collections import defaultdict
import numpy as np
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt


comments_file = "../dataset/CommentsJan2017.csv"
data = pd.read_csv(comments_file, usecols=["commentBody", "articleID"]).head(1000)


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    return re.sub(r'\s+', ' ', text).strip() 


data["cleaned_commentBody"] = data["commentBody"].apply(clean_text)


analyzer = SentimentIntensityAnalyzer()

def sentiment_scores(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment["neg"], sentiment["neu"], sentiment["pos"]


tqdm.pandas(desc="Sentiment Analysis")
data[["neg", "neu", "pos"]] = data["cleaned_commentBody"].progress_apply(lambda x: pd.Series(sentiment_scores(x)))


nlp = spacy.load("en_core_web_sm")


def extract_meaningful_words(text):
    doc = nlp(text)

    allowed_pos_tags = {"NOUN", "VERB", "ADJ", "ADV"}
    return " ".join([token.text for token in doc if token.pos_ in allowed_pos_tags])


tqdm.pandas(desc="Extracting Keywords")
data["keywords"] = data["cleaned_commentBody"].progress_apply(extract_meaningful_words)


print(data[["commentBody", "cleaned_commentBody", "neg", "neu", "pos", "keywords"]].head())




# --- GNN model


G = nx.Graph()


cooccurrence_dict = defaultdict(lambda: {"neg": 0, "neu": 0, "pos": 0})


for _, row in data.iterrows():
    keywords = row["keywords"].split()
    neg, neu, pos = row["neg"], row["neu"], row["pos"]


    for word1, word2 in combinations(keywords, 2):
        cooccurrence_dict[(word1, word2)]["neg"] += neg
        cooccurrence_dict[(word1, word2)]["neu"] += neu
        cooccurrence_dict[(word1, word2)]["pos"] += pos


for (word1, word2), sentiment_weights in cooccurrence_dict.items():
    total_weight = sentiment_weights["neg"] + sentiment_weights["neu"] + sentiment_weights["pos"]
    G.add_edge(word1, word2, weight=total_weight)


node_features = defaultdict(lambda: np.zeros(3))


for _, row in data.iterrows():
    keywords = row["keywords"].split()
    neg, neu, pos = row["neg"], row["neu"], row["pos"]

    for word in keywords:
        node_features[word] += np.array([neg, neu, pos])


for _, row in data.iterrows():
    keywords = row["keywords"].split()
    for word in keywords:
        if word not in G:
            G.add_node(word)


for word, features in node_features.items():
    if word in G:
        G.nodes[word]["features"] = features
    else:
        print(f"Warning: {word} not found in graph nodes.")


subgraph = G.subgraph(list(G.nodes)[:20])


pos = nx.spring_layout(subgraph)
edge_labels = nx.get_edge_attributes(subgraph, "weight")

plt.figure(figsize=(12, 8))
nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color="skyblue")
nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels)
plt.title("Sample of Word Co-occurrence Graph with Aggregated Sentiment Weights (Filtered)")
plt.show()


# output

def infer_related_words(graph, keywords, top_n=10):
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


input_keywords = ["tax", "policy"]

related_words = infer_related_words(G, input_keywords, top_n=10)
print(f"Related words for {input_keywords} are: {related_words}")
