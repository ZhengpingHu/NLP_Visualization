import os
os.environ["PYTHONWARNINGS"] = "ignore"
import torch
import torch.nn as nn 
import torch.nn.functional as F
import plotly.graph_objects as go
import networkx as nx
import pickle
from torch_geometric.nn import GCNConv

input_dim = 100
hidden_dim = 64
output_dim = 3

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

model = GNNModel(input_dim, hidden_dim, output_dim)
# model.load_state_dict(torch.load('../Data_Analysis_Demo/models/gnn_model.pth'))
model.load_state_dict(torch.load('./Data_Analysis_Demo/models/gnn_model.pth'))

model.eval()

data_pg = torch.load('./Data_Analysis_Demo/models/graph_data.pth')


with open('./Data_Analysis_Demo/models/word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

with open('./Data_Analysis_Demo/models/index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

def GNN_result(model, data_pg, input_keywords, sentiment_direction, top_n=10):
    sentiment_idx = {"neg": 0, "neu": 1, "pos": 2}
    idx = sentiment_idx.get(sentiment_direction.lower())
    if idx is None:
        raise ValueError("Invalid sentiment direction. Choose from 'neg', 'neu', 'pos'.")

    input_indices = [word_to_index[word] for word in input_keywords if word in word_to_index]
    if not input_indices:
        return [], None

    with torch.no_grad():
        embeddings = model(data_pg)
        input_embeddings = embeddings[input_indices]

        scores = embeddings[:, idx]
        input_score = input_embeddings[:, idx].mean().item()
        sentiment_differences = torch.abs(scores - input_score)

        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_input_embeddings = F.normalize(input_embeddings, p=2, dim=1)
        mean_input_embedding = normalized_input_embeddings.mean(dim=0)
        cosine_similarities = torch.matmul(normalized_embeddings, mean_input_embedding)

        combined_scores = cosine_similarities - sentiment_differences

        for idx_input in input_indices:
            combined_scores[idx_input] = float('-inf')

        sorted_indices = torch.argsort(combined_scores, descending=True)[:top_n]
        related_words = [index_to_word[idx.item()] for idx in sorted_indices]

    fig = generate_3d_graph(related_words, input_keywords, sentiment_direction)
    html_path = "../static/related_graph.html"
    fig.write_html(html_path)

    return related_words, html_path

def generate_3d_graph(related_words, input_keywords, sentiment_direction):
    sub_nodes = set(related_words + input_keywords)
    subgraph = nx.Graph()
    for edge in data_pg.edge_index.t().tolist():
        node_u = index_to_word[edge[0]]
        node_v = index_to_word[edge[1]]
        if node_u in sub_nodes and node_v in sub_nodes:
            subgraph.add_edge(node_u, node_v)

    pos = nx.spring_layout(subgraph, dim=3, seed=42)
    sentiment_idx = {"neg": 0, "neu": 1, "pos": 2}
    node_colors = [
        data_pg.y[word_to_index[node]][sentiment_idx[sentiment_direction.lower()]].item()
        for node in subgraph.nodes()
    ]

    edge_x, edge_y, edge_z = [], [], []
    for edge in subgraph.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, line=dict(width=2, color='gray'), mode='lines')
    node_x, node_y, node_z, text = [], [], [], []
    for node in subgraph.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        text.append(node)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode='markers+text',
        text=text, textposition="top center", hoverinfo='text',
        marker=dict(size=10, color=node_colors, colorscale='Greens', line_width=1)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title=f'3D Graph of {input_keywords} ({sentiment_direction})'))
    return fig
