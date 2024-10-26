import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations
import os
import scipy.sparse as sp
import joblib
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

data_file = "../dataset/ProcessedCommentsAll.csv"
data = pd.read_csv(data_file)

weight_threshold = 0.7
num_partitions = 10
num_workers = max(os.cpu_count() - 2, 8)

data_splits = np.array_split(data, num_partitions)


def process_batch(data_batch):
    local_cooccurrence_dict = defaultdict(lambda: {"neg": 0, "neu": 0, "pos": 0})
    local_node_features = defaultdict(lambda: np.zeros(3))

    for _, row in data_batch.iterrows():
        keywords = str(row["keywords"]).split() if pd.notna(row["keywords"]) else []
        neg, neu, pos = row["neg"], row["neu"], row["pos"]

        for word1, word2 in combinations(keywords, 2):
            local_cooccurrence_dict[(word1, word2)]["neg"] += neg
            local_cooccurrence_dict[(word1, word2)]["neu"] += neu
            local_cooccurrence_dict[(word1, word2)]["pos"] += pos

        for word in keywords:
            local_node_features[word] += np.array([neg, neu, pos])

    return local_cooccurrence_dict, local_node_features

global_cooccurrence_dict = defaultdict(lambda: {"neg": 0, "neu": 0, "pos": 0})
global_node_features = defaultdict(lambda: np.zeros(3))

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(process_batch, batch) for batch in data_splits]
    for future in tqdm(as_completed(futures), total=num_partitions, desc="Processing Batches"):
        local_cooccurrence_dict, local_node_features = future.result()

        for (word1, word2), sentiment_weights in local_cooccurrence_dict.items():
            global_cooccurrence_dict[(word1, word2)]["neg"] += sentiment_weights["neg"]
            global_cooccurrence_dict[(word1, word2)]["neu"] += sentiment_weights["neu"]
            global_cooccurrence_dict[(word1, word2)]["pos"] += sentiment_weights["pos"]

        for word, features in local_node_features.items():
            global_node_features[word] += features

G = nx.Graph()
for (word1, word2), sentiment_weights in global_cooccurrence_dict.items():
    total_weight = sentiment_weights["neg"] + sentiment_weights["neu"] + sentiment_weights["pos"]
    if total_weight >= weight_threshold:
        G.add_edge(word1, word2, weight=total_weight)

for word, features in global_node_features.items():
    if word in G and G.degree[word] > 1:
        G.nodes[word]["features"] = features

adj_matrix = nx.to_scipy_sparse_matrix(G, weight="weight", format="csr")

model_dir = "../models/"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "sentiment_cooccurrence_graph_compressed.pkl.gz")

with gzip.open(model_path, "wb") as f:
    joblib.dump((G, adj_matrix), f)

print(f"Compressed model saved to {model_path}")
