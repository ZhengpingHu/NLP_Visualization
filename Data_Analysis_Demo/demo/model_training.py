import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations
import os
import joblib
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

MAX_CPUS = 10  
MIN_WEIGHT = 0.7
BATCH_SIZE = 3000
num_processes = min(cpu_count() - 2, MAX_CPUS)

data_file = "../dataset/ProcessedCommentsAll.csv"
data = pd.read_csv(data_file)
temp_dir = "../temp/"
os.makedirs(temp_dir, exist_ok=True)



# 定义默认情感字典函数
def default_sentiment_dict():
    return {"neg": 0, "neu": 0, "pos": 0}

def process_batch_to_temp(args):
    data_batch, batch_num = args
    cooccurrence_dict = defaultdict(default_sentiment_dict)
    for _, row in data_batch.iterrows():
        keywords = str(row["keywords"]).split() if pd.notna(row["keywords"]) else []
        neg, neu, pos = row["neg"], row["neu"], row["pos"]

        for word1, word2 in combinations(keywords, 2):
            cooccurrence_dict[(word1, word2)]["neg"] += neg
            cooccurrence_dict[(word1, word2)]["neu"] += neu
            cooccurrence_dict[(word1, word2)]["pos"] += pos

    batch_file = os.path.join(temp_dir, f"batch_{batch_num}.pkl")
    joblib.dump(dict(cooccurrence_dict), batch_file)

def load_and_aggregate_batches():
    final_cooccurrence_dict = defaultdict(default_sentiment_dict)
    batch_files = [f for f in os.listdir(temp_dir) if f.startswith("batch_")]

    for batch_file in tqdm(batch_files, desc="Aggregating batches"):
        batch_path = os.path.join(temp_dir, batch_file)
        batch_cooccurrence = joblib.load(batch_path)
        for key, sentiment_weights in batch_cooccurrence.items():
            final_cooccurrence_dict[key]["neg"] += sentiment_weights["neg"]
            final_cooccurrence_dict[key]["neu"] += sentiment_weights["neu"]
            final_cooccurrence_dict[key]["pos"] += sentiment_weights["pos"]

    return final_cooccurrence_dict

def main():
    total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    data_batches = [(data.iloc[start:start + BATCH_SIZE], batch_num) for batch_num, start in enumerate(range(0, len(data), BATCH_SIZE))]

    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_batch_to_temp, data_batches), total=total_batches, desc="Processing batches"))

    G = nx.Graph()
    final_cooccurrence_dict = load_and_aggregate_batches()

    for (word1, word2), sentiment_weights in final_cooccurrence_dict.items():
        total_weight = sentiment_weights["neg"] + sentiment_weights["neu"] + sentiment_weights["pos"]
        if total_weight >= MIN_WEIGHT:
            G.add_edge(word1, word2, weight=total_weight)

    model_dir = "../models/"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sentiment_cooccurrence_graph.pkl")
    joblib.dump(G, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
