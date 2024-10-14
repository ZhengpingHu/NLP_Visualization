from flask import Flask, render_template, jsonify, request, stream_with_context, Response
import requests
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit-poem', methods=['POST'])
def submit_poem():
    prompt = request.form.get('poem')
    if prompt:
        return Response(stream_with_context(call_llama_api_stream(prompt)), content_type='text/html')
    return render_template('index.html', error="Please enter a valid poem.")

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
                yield "<html><body><h1>The idea from llama 3.2</h1><h3>Your question:</h3><p>{}</p><h3>Result:</h3><p>".format(prompt)
                for chunk in response.iter_lines():
                    if chunk:
                        json_chunk = json.loads(chunk.decode('utf-8'))
                        message_part = json_chunk.get('message', {}).get('content', '')
                        yield message_part
                yield "</p><a href='/'>Go back</a></body></html>"
            else:
                yield "<p>Error: Unable to connect to llama API</p>"
    except requests.exceptions.RequestException as e:
        yield f"<p>Error: {e}</p>"

if __name__ == '__main__':
    # use ifconfig to check the local ip address
    app.run(host='192.168.0.246', port=6612, debug=True)
