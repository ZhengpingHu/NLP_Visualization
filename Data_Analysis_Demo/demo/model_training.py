import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations
import os
import scipy.sparse as sp
import joblib
import gzip

data_file = "../dataset/ProcessedCommentsAll.csv"
data = pd.read_csv(data_file)

G = nx.Graph()
cooccurrence_dict = defaultdict(lambda: {"neg": 0, "neu": 0, "pos": 0})

weight_threshold = 0.5

for _, row in data.iterrows():
    keywords = str(row["keywords"]).split() if pd.notna(row["keywords"]) else []
    neg, neu, pos = row["neg"], row["neu"], row["pos"]

    for word1, word2 in combinations(keywords, 2):
        cooccurrence_dict[(word1, word2)]["neg"] += neg
        cooccurrence_dict[(word1, word2)]["neu"] += neu
        cooccurrence_dict[(word1, word2)]["pos"] += pos

for (word1, word2), sentiment_weights in cooccurrence_dict.items():
    total_weight = sentiment_weights["neg"] + sentiment_weights["neu"] + sentiment_weights["pos"]
    if total_weight >= weight_threshold:
        G.add_edge(word1, word2, weight=total_weight)

node_features = defaultdict(lambda: np.zeros(3))
for _, row in data.iterrows():
    keywords = str(row["keywords"]).split() if pd.notna(row["keywords"]) else []
    neg, neu, pos = row["neg"], row["neu"], row["pos"]

    for word in keywords:
        node_features[word] += np.array([neg, neu, pos])

for word, features in node_features.items():
    if word in G and G.degree[word] > 1:
        G.nodes[word]["features"] = features

adj_matrix = nx.to_scipy_sparse_matrix(G, weight="weight", format="csr")

model_dir = "../models/"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "sentiment_cooccurrence_graph_compressed.pkl.gz")

with gzip.open(model_path, "wb") as f:
    joblib.dump((G, adj_matrix), f)

print(f"Compressed model saved to {model_path}")
