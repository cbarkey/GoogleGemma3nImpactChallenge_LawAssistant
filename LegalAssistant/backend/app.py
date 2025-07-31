from flask import Flask, request, jsonify
from flask_cors import CORS
import model
from model.RAGPipeline import RAGPipelineClass

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

# Load the RAG System
# qa_chain = load_rag_chain()
RAG = RAGPipelineClass()

@app.route('/chat', methods=['POST'])
def chat():
    # Take in the user input from the application
    data = request.get_json()
    user_message = data.get('message', '')
    print("Received:", user_message)

    
    # response_text = f"You said: {user_message}"

    # Run the user input through the RAG Pipeline to get an answer
    result = RAGPipelineClass.get_answer(user_message)
    
    print("Result:", result["answer"])
    return jsonify({"response": result})

if __name__ == '__main__':
    app.run(debug=True)
