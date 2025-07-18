from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    print("Received:", user_message)

    # TODO: Replace this with actual AI logic later
    response_text = f"You said: {user_message}"

    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)
