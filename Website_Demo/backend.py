import os
import json
import requests
import torch
import torch.nn.functional as F
import pickle
import plotly
import plotly.graph_objects as go
import networkx as nx
import torch.nn as nn
from torch_geometric.nn import GCNConv
from flask import Flask, render_template, request, jsonify, url_for, redirect, Response, stream_with_context
from celery import Celery
from celery.result import AsyncResult
import markdown
from markupsafe import Markup

app = Flask(__name__)


# Setup Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'  # by using redis 
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


# import the GNN model from GNN.py file. Its a simple reuse. Source code see Data_Analysis_Demo folder.
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

gnn_model = None
data_pg = None
word_to_index = None
index_to_word = None

def load_gnn_model():
    global gnn_model, data_pg, word_to_index, index_to_word
    if gnn_model is None:
        input_dim = 100
        hidden_dim = 64
        output_dim = 3

        model = GNNModel(input_dim, hidden_dim, output_dim)
        model_path = '/media/lunar/drive/Git/NLP_Visualization/Data_Analysis_Demo/models/gnn_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        data_path = '/media/lunar/drive/Git/NLP_Visualization/Data_Analysis_Demo/models/graph_data.pth'
        data_pg = torch.load(data_path, map_location=torch.device('cpu'))

        with open('/media/lunar/drive/Git/NLP_Visualization/Data_Analysis_Demo/models/word_to_index.pkl', 'rb') as f:
            word_to_index = pickle.load(f)
        with open('/media/lunar/drive/Git/NLP_Visualization/Data_Analysis_Demo/models/index_to_word.pkl', 'rb') as f:
            index_to_word = pickle.load(f)

        gnn_model = model


# also a reuse from GNN model, the function would get the word and snetiment direction from submit.html, send to GNN model.
def find_related_words(model, data_pg, word_to_index, index_to_word, input_words, sentiment_direction, top_n=10):
    sentiment_idx = {"negative": 0, "neutral": 1, "positive": 2}
    idx = sentiment_idx.get(sentiment_direction.lower())
    if idx is None:
        raise ValueError("Invalid sentiment direction. Choose from 'negative', 'neutral', 'positive'.")

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

        # Calculate similarity
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


# Get the graph from the GNN model, absolutely graph algorithm.
def visualize_graph(related_words, input_keywords, sentiment_direction):
    # Combine the input word and related words
    sub_nodes = set(related_words + input_keywords)
    if not sub_nodes:
        print("No nodes to visualize.")
        return {}

    subgraph = nx.Graph()
    for edge in data_pg.edge_index.t().tolist():
        node_u = index_to_word[edge[0]]
        node_v = index_to_word[edge[1]]
        if node_u in sub_nodes and node_v in sub_nodes:
            subgraph.add_edge(node_u, node_v)

    # Get Node Position
    pos = nx.spring_layout(subgraph, dim=3, seed=42)

    # Get Nodes Sentiment
    sentiment_idx = {"negative": 0, "neutral": 1, "positive": 2}
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
            colorscale='Viridis',
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
    # Convert the plotly figure to JSON for rendering in the template
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

# Route for index page.
@app.route('/')
def home():
    return render_template('index.html')


# Route for submit page.
@app.route('/submit')
def submit_page():
    return render_template('submit.html')

# Route for explormentray page.
@app.route('/explor')
def explor_page():
    content = [
        {'type': 'image', 'image': 'static/1.png', 'title': 'Exploratory analysis: 2d distribution among comment\'s different emotions', 'text': ' '},
        {'type': 'plot', 'plot': 'static/2_3D_Sentiment_Score_Distribution.html', 'title': 'Exploratory analysis: 3D distribution among comment\'s different emotions', 'text': 'Overall, this graph shows that the overall sentiment tendency of comments is more moderate and dominated by neutral sentiment. This distribution may reflect commenters\' neutral or mild reactions to the content of the article, rather than extreme emotions.'},
        {'type': 'image', 'image': 'static/3.png', 'title': 'Exploratory analysis: 2d distribution among comment\'s different emotions. In specific comments.', 'text': ' '},
        {'type': 'plot', 'plot': 'static/4_3D_Sentiment_Score_Distribution.html', 'title': 'Exploratory analysis: 3d distribution among comment\'s different emotions. In specific comments.', 'text': 'Overall, this graph further confirms that the overall sentiment of the comments is skewed towards neutral and mildly positive sentiment, with only a small number of comments showing significant negative sentiment. This may reflect a high level of acceptance of the topic by the commenters, or that the topic discussed did not elicit a strong emotional response.'},
        {'type': 'image', 'image': 'static/5.png', 'title': 'Word Cloud with Custom Color Gradient for Positive Sentiment', 'text': 'Positive Sentiment Word Cloud: Some common positive words such as “thank”, “great”, “good”, “love”, “excellent”, and “beautiful” can be seen in the Positive Sentiment Word Cloud, which express positive emotions of the readers.'},
        {'type': 'image', 'image': 'static/6.png', 'title': 'Word Cloud with Custom Color Gradient for Negative Sentiment', 'text': 'Negative Sentiment Word Cloud: the negative sentiment word cloud on the other hand highlights words such as “trump”, “war”, “wrong”, “bad”, and “hate”, showing the negative sentiments of the readers.'},
        {'type': 'image', 'image': 'static/7.png', 'title': 'Word Cloud with Custom Color Gradient for Neutral Sentiment', 'text': 'Neutral emotion word cloud: The neutral emotion word cloud mainly contains some descriptive or informative words, such as “people”, “have”, “more”, “just”, “know”, “country”, etc. These words are usually used to describe the facts or make neutral points.'}
    ]
    return render_template('explor.html', content=content)


# Route for visual analysis page.
@app.route('/visual')
def visual_page():
    items = [
        {'image': 'static/8.jpg', 'title': 'Keyword word cloud of the year based on sentiment intensity', 'text': 'In 2017, keyword sentiment reflected political controversies like Trump and Russiagate, while in 2018, focus shifted towards social issues like gun violence, with political keywords becoming more neutral and analytical.'},
        {'image': 'static/9.jpg', 'title': 'Annual Keyword Word Cloud Based on Number of Comments', 'text': 'In 2017, comments focused on political issues like Trump and policy controversies, while in 2018, the focus shifted to social issues like gun violence, with an increase in negatively charged discussions.'},
        {'image': 'static/10.png', 'title': '2D Keyword Co-occurrence Network Analysis', 'text': 'The dense structure reveals interconnected discussions on themes like "election," "law," and "democracy," showing the public\'s holistic interest in fairness, transparency, and governance mechanisms.'}
    ]
    plots = [
        'static/11_3D_Keyword_Cooccurrence_Network.html'
    ]
    return render_template('visual.html', items=items, plots=plots)

# Route for conclusion page.
@app.route('/conc')
def conclusion_page():
    return render_template('conclusion.html')

# Route for submit-query, the middle page for submit.html.
@app.route('/submit-query', methods=['POST'])
def submit_query():
    keyword = request.form.get('keyword')
    sentiment = request.form.get('sentiment')

    if keyword and sentiment:
        task = process_model_task.apply_async(args=[keyword, sentiment])
        return redirect(url_for('task_status', task_id=task.id))
    else:
        return render_template('submit.html', error="Please enter a valid keyword and select a sentiment.")

# Route for processing page, the middle status.
@app.route('/task-status/<task_id>')
def task_status(task_id):
    return render_template('processing.html', task_id=task_id)

@app.route('/task-result/<task_id>')
def task_result(task_id):
    task = AsyncResult(task_id, app=celery)
    if task.state == 'SUCCESS':
        result = task.get()
        if 'error' in result:
            return render_template('submit.html', error=result['error'])
        graphJSON = json.loads(result['graphJSON'])
        return render_template('result.html', prompt=result['prompt'], graphJSON=graphJSON)
    elif task.state == 'FAILURE':
        return "An error occurred during processing. Please try again."
    else:
        return redirect(url_for('task_status', task_id=task_id))

# Route for celery-status, we must have a celery for polling or we would never get the result because of the delay.
@app.route('/celery-status/<task_id>')
def celery_task_status(task_id):
    task = AsyncResult(task_id, app=celery)
    response = {'state': task.state}
    if task.state == 'PENDING':
        response['status'] = 'Pending...'
    elif task.state == 'PROGRESS':
        response['status'] = 'In progress...'
        response['meta'] = task.info
    elif task.state == 'SUCCESS':
        response['status'] = 'Task completed!'
    elif task.state == 'FAILURE':
        response['status'] = 'Task failed.'
        response['error'] = str(task.info)
    else:
        response['status'] = str(task.info)
    return jsonify(response)

# Route for stream display of LLM (llama3.2)
@app.route('/stream')
def stream():
    prompt = request.args.get('prompt')
    if not prompt:
        return "Prompt is missing.", 400
    return Response(stream_with_context(call_llama_api_stream(prompt)), content_type='text/event-stream')

# Regular llama API by json.
def call_llama_api_stream(prompt):
    api_url = "http://localhost:11434/api/chat"
    data = {
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }
    headers = {"Content-Type": "application/json"}
    try:
        with requests.post(api_url, data=json.dumps(data), headers=headers, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        json_data = json.loads(decoded_line)
                        message_part = json_data.get('message', {}).get('content', '')
                        yield f"data: {message_part}\n\n"
            else:
                yield "data: Error: Unable to connect to Llama API\n\n"
    except requests.exceptions.RequestException as e:
        yield f"data: Error: {e}\n\n"

@celery.task(bind=True)
def process_model_task(self, keyword, sentiment):
    try:
        # model loaded
        load_gnn_model()
        print(f"Processing task with keyword: {keyword}, sentiment: {sentiment}")
        self.update_state(state='PROGRESS', meta={'step': 'Parsing input keywords'})
        input_keywords = keyword.strip().split()
        print(f"Input keywords: {input_keywords}")

        self.update_state(state='PROGRESS', meta={'step': 'Finding related words'})
        related_words = find_related_words(
            gnn_model, data_pg, word_to_index, index_to_word, input_keywords, sentiment, top_n=10)
        print(f"Related words: {related_words}")

        if not related_words:
            print("No related words found.")
            return {'error': "No related words found for the given keyword."}

        self.update_state(state='PROGRESS', meta={'step': 'Preparing prompt for LLM'})
        sentiment_mapping = {
            "Positive": "positive",
            "Negative": "negative",
            "Neutral": "neutral",
        }
        sentiment_en = sentiment_mapping.get(sentiment, sentiment)

        # Prompt part
        prompt = (
            f"You are an experienced writing instructor. Based on the following information, please provide a detailed writing guide in English.\n\n"
            f"User Input Keywords: {', '.join(input_keywords)}\n"
            f"Selected Sentiment: {sentiment_en}\n"
            f"GNN Model Suggested Keywords: {', '.join(related_words)}\n\n"
            f"Please create an outline for a piece of writing that incorporates the above keywords and reflects the selected sentiment. The outline should include sections, bullet points, and suggestions. **Do not write the full text.**"
        )
        print(f"Prompt: {prompt}")

        self.update_state(state='PROGRESS', meta={'step': 'Generating graph visualization'})
        graph_json = visualize_graph(related_words, input_keywords, sentiment)
        print("Graph visualization generated.")

        print("Task completed successfully.")
        return {'prompt': prompt, 'graphJSON': graph_json}
    except Exception as e:
        print(f"Error in process_model_task: {e}")
        self.update_state(state='FAILURE', meta={'exc_message': str(e)})
        raise e

if __name__ == '__main__':
    app.run(host='192.168.2.96', port=6612, debug=False)
