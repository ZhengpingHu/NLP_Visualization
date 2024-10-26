from flask import Flask, render_template, request, Response, stream_with_context
import requests
import json
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit')
def submit_page():
    return render_template('submit.html')

@app.route('/team')
def team_page():
    return render_template('team_intro.html')

@app.route('/specific')
def specific_page():
    plot_html = 'static/myplot.html'
    return render_template('specific.html', plot_html=plot_html)

@app.route('/wordcloud')
def worldcloud_page():
    plot_png = 'static/myplot.png'
    return render_template('wordcloud.html', plot_png=plot_png)

@app.route('/submit-poem', methods=['POST'])
def submit_poem():
    poem = request.form.get('poem')
    if poem:
        return render_template('result.html', poem=poem)
    return render_template('submit.html', error="Please enter a valid poem.")

@app.route('/stream/<poem>')
def stream(poem):
    return Response(stream_with_context(call_llama_api_stream(poem)), content_type='text/event-stream')

def call_llama_api_stream(prompt):
    api_url = "http://localhost:11434/api/chat"
    data = {
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    headers = {"Content-Type": "application/json"}
    try:
        with requests.post(api_url, data=json.dumps(data), headers=headers, stream=True) as response:
            if response.status_code == 200:
                for chunk in response.iter_lines():
                    if chunk:
                        json_chunk = json.loads(chunk.decode('utf-8'))
                        message_part = json_chunk.get('message', {}).get('content', '')
                        yield f"data: {message_part}\n\n"
            else:
                yield "data: Error: Unable to connect to Llama API\n\n"
    except requests.exceptions.RequestException as e:
        yield f"data: Error: {e}\n\n"

if __name__ == '__main__':
    app.run(host='192.168.0.246', port=6612, debug=True)
