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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from gensim.models import Word2Vec

# 1. 数据预处理

comments_file = "../dataset/CommentsJan2017.csv"
data = pd.read_csv(comments_file, usecols=["commentBody", "articleID"]).head(2000)  # 您可以调整读取的数据量

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    return re.sub(r'\s+', ' ', text).strip()

data["cleaned_commentBody"] = data["commentBody"].apply(clean_text)

# 情感分析
analyzer = SentimentIntensityAnalyzer()

def sentiment_scores(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment["neg"], sentiment["neu"], sentiment["pos"]

tqdm.pandas(desc="Sentiment Analysis")
data[["neg", "neu", "pos"]] = data["cleaned_commentBody"].progress_apply(lambda x: pd.Series(sentiment_scores(x)))

# 关键词提取
nlp = spacy.load("en_core_web_sm")

def extract_meaningful_words(text):
    doc = nlp(text)
    allowed_pos_tags = {"NOUN", "VERB", "ADJ", "ADV", "X"}
    return " ".join([token.text.lower() for token in doc if token.pos_ in allowed_pos_tags])

tqdm.pandas(desc="Extracting Keywords")
data["keywords"] = data["cleaned_commentBody"].progress_apply(extract_meaningful_words)

print(data[["commentBody", "cleaned_commentBody", "neg", "neu", "pos", "keywords"]].head())

# 2. 构建共现图

G = nx.Graph()

cooccurrence_dict = defaultdict(lambda: {"neg": 0, "neu": 0, "pos": 0})

for _, row in data.iterrows():
    keywords = row["keywords"].split()
    neg, neu, pos = row["neg"], row["neu"], row["pos"]

    for word1, word2 in combinations(keywords, 2):
        if word1 != word2:
            cooccurrence_dict[(word1, word2)]["neg"] += neg
            cooccurrence_dict[(word1, word2)]["neu"] += neu
            cooccurrence_dict[(word1, word2)]["pos"] += pos

for (word1, word2), sentiment_weights in cooccurrence_dict.items():
    total_weight = sentiment_weights["neg"] + sentiment_weights["neu"] + sentiment_weights["pos"]
    G.add_edge(word1, word2, weight=total_weight)

# 3. 准备节点特征和标签

# (a) 生成节点特征（词向量）

# 获取所有关键词的集合
all_keywords = set()
for keywords in data["keywords"]:
    all_keywords.update(keywords.split())

# 训练Word2Vec模型
sentences = [keywords.split() for keywords in data["keywords"]]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 为每个节点生成词向量特征
node_features = {}
for word in all_keywords:
    node_features[word] = word2vec_model.wv[word]

# (b) 为每个节点生成情感标签

# 计算每个词的累积情感得分
node_sentiments = defaultdict(lambda: np.zeros(3))
for _, row in data.iterrows():
    keywords = row["keywords"].split()
    neg, neu, pos = row["neg"], row["neu"], row["pos"]

    for word in keywords:
        node_sentiments[word] += np.array([neg, neu, pos])

# 对情感得分进行归一化
for word in node_sentiments:
    total = node_sentiments[word].sum()
    if total > 0:
        node_sentiments[word] /= total
    else:
        node_sentiments[word] = np.array([0, 0, 0])

# 4. 准备图数据用于GNN

# 将节点特征和标签添加到图中
for node in G.nodes():
    G.nodes[node]['x'] = node_features.get(node, np.zeros(100))  # 如果没有特征，使用零向量
    G.nodes[node]['y'] = node_sentiments.get(node, np.zeros(3))  # 如果没有情感得分，使用零向量

# 将 NetworkX 图转换为 PyTorch Geometric 的数据对象
data_pg = from_networkx(G)

# 转换节点特征和标签为 tensor，先将列表转换为 numpy.ndarray
data_pg.x = torch.tensor(np.array([G.nodes[node]['x'] for node in G.nodes()]), dtype=torch.float)
data_pg.y = torch.tensor(np.array([G.nodes[node]['y'] for node in G.nodes()]), dtype=torch.float)

# 5. 定义GNN模型

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return x

# 6. 训练模型

input_dim = data_pg.num_node_features
hidden_dim = 64
output_dim = 3  # 情感得分维度

model = GNNModel(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data_pg)
    loss = criterion(out, data_pg.y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 7. prediction and output

# create the word to ordering
word_to_index = {word: idx for idx, word in enumerate(G.nodes())}
index_to_word = {idx: word for word, idx in word_to_index.items()}

def find_related_words(model, data_pg, input_words, sentiment_direction, top_n=10):
    sentiment_idx = {"neg": 0, "neu": 1, "pos": 2}
    idx = sentiment_idx.get(sentiment_direction.lower())
    if idx is None:
        raise ValueError("Invalid sentiment direction. Choose from 'neg', 'neu', 'pos'.")

    input_indices = [word_to_index[word] for word in input_words if word in word_to_index]
    if not input_indices:
        print("No input words found in the graph.")
        return []

    model.eval()
    with torch.no_grad():
        embeddings = model(data_pg)
        input_embeddings = embeddings[input_indices]

        # have the sum score in all emotions
        scores = embeddings[:, idx]

        # calculate the avg score in each emotion
        input_score = input_embeddings[:, idx].mean().item()

        # calculate the differences, less differences means related.
        differences = torch.abs(scores - input_score)

        # skip the word itself
        for idx_input in input_indices:
            differences[idx_input] = float('inf')

        # get the index of most related words
        sorted_indices = torch.argsort(differences)[:top_n]

        # back to the word itself
        related_words = [index_to_word[idx.item()] for idx in sorted_indices]

    return related_words

# Example

input_keywords = ["trump"]  # Modify
sentiment_direction = "pos"  # 可选择 "neg", "neu", "pos"

related_words = find_related_words(model, data_pg, input_keywords, sentiment_direction, top_n=10)
print(f"Related words for {input_keywords} in {sentiment_direction} sentiment are: {related_words}")
