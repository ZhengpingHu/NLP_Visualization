import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle
import networkx as nx
import plotly.graph_objects as go

# Define GNN model.
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

# Load training model
input_dim = 100
hidden_dim = 64
output_dim = 3

model = GNNModel(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('../models/gnn_model.pth'))
model.eval()

data_pg = torch.load('../models/graph_data.pth')

with open('../models/word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)
with open('../models/index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)


def find_related_words(model, data_pg, input_words, sentiment_direction, top_n=10):
    sentiment_idx = {"neg": 0, "neu": 1, "pos": 2}
    idx = sentiment_idx.get(sentiment_direction.lower())
    if idx is None:
        raise ValueError("Invalid sentiment direction. Choose from 'neg', 'neu', 'pos'.")

    input_indices = [word_to_index[word] for word in input_words if word in word_to_index]
    if not input_indices:
        print("No input words found in the graph.")
        return []

    with torch.no_grad():
        embeddings = model(data_pg)
        input_embeddings = embeddings[input_indices]

        # Calculate sentiment score
        scores = embeddings[:, idx]
        input_score = input_embeddings[:, idx].mean().item()
        sentiment_differences = torch.abs(scores - input_score)

        # cauculate the similar
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_input_embeddings = F.normalize(input_embeddings, p=2, dim=1)
        mean_input_embedding = normalized_input_embeddings.mean(dim=0)
        cosine_similarities = torch.matmul(normalized_embeddings, mean_input_embedding)

        # Combine total sentiments.
        combined_scores = cosine_similarities - sentiment_differences

        # Ignore input word itself
        for idx_input in input_indices:
            combined_scores[idx_input] = float('-inf')

        # Most related words.
        sorted_indices = torch.argsort(combined_scores, descending=True)[:top_n]
        related_words = [index_to_word[idx.item()] for idx in sorted_indices]

    return related_words

# User input.
input_keywords = input("Please input the single keyword: ").strip().split()
sentiment_direction = input("Choose the sentiment:(neg, neu, pos)：").strip()

# Get keyword.
related_words = find_related_words(model, data_pg, input_keywords, sentiment_direction, top_n=10)
print(f"For {input_keywords} word, at {sentiment_direction} sentiment, related words are：{related_words}")

# 3D Graph
def visualize_graph(related_words, input_keywords, sentiment_direction):
    # Combine the input word and related words
    sub_nodes = set(related_words + input_keywords)
    subgraph = nx.Graph()
    for edge in data_pg.edge_index.t().tolist():
        node_u = index_to_word[edge[0]]
        node_v = index_to_word[edge[1]]
        if node_u in sub_nodes and node_v in sub_nodes:
            subgraph.add_edge(node_u, node_v)

    # Get Node Position
    pos = nx.spring_layout(subgraph, dim=3, seed=42)

    # Get Nodes Sentiment
    sentiment_idx = {"neg": 0, "neu": 1, "pos": 2}
    node_colors = []
    for node in subgraph.nodes():
        idx = word_to_index[node]
        sentiment_score = data_pg.y[idx][sentiment_idx[sentiment_direction.lower()]].item()
        node_colors.append(sentiment_score)

    # Create the 3D Graph
    edge_x = []
    edge_y = []
    edge_z = []

    for edge in subgraph.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_z = []
    text = []
    for node in subgraph.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        text.append(node)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=10,
            color=node_colors,
            colorscale='Greens',
            colorbar=dict(title='Sentiment Score'),
            line_width=1))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Keyword: {input_keywords} with {sentiment_direction} sentiment graph',
                        showlegend=False,
                        scene=dict(
                            xaxis=dict(showbackground=False),
                            yaxis=dict(showbackground=False),
                            zaxis=dict(showbackground=False)
                        ),
                        margin=dict(b=20, l=5, r=5, t=40)
                    ))
    fig.show()

# Visualization
visualize_graph(related_words, input_keywords, sentiment_direction)
