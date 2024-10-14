import requests
import json

api_url = "http://localhost:11434/api/chat"

data = {
    "model": "llama3.2",
    "messages": [
        { "role": "user", "content": "Please assume that you are an editor of the New York Time and generate a new brief writing outline using Trump, Tax, technique as keywords and positive sentiment." }
    ]
}


response = requests.post(api_url, data=json.dumps(data), headers={"Content-Type": "application/json"})


for line in response.iter_lines():
    if line:
        try:
            message_data = json.loads(line.decode('utf-8'))
            message = message_data.get('message', {})
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            print(f"{content}", end='')
        except json.JSONDecodeError:
            print("Error decoding JSON line:", line)
print('')
