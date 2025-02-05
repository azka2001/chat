from flask import Flask, render_template, request, jsonify
from llm_module import create_vector_store, generate_response
import logging
import os
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)  # Add logging

CORS(app)

app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

vector_store = None  # Initialize global variable

@app.route('/')
def main():
    print("Main route accessed")
    print("Current working directory:", os.getcwd())
    print("Contents of current directory:", os.listdir())
    print("Contents of templates directory:", os.listdir('templates'))
    return render_template('main.html')

@app.route('/chatbot', methods=['GET'])
def chatbot():
    global vector_store
    try:
        website = request.args.get('website')

        if website:
            # Create a new vector store for the new website
            vector_store = create_vector_store(website)

        return render_template('chatbot.html')
    except Exception as e:
        logging.error(f"Error in chatbot route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    global vector_store
    try:
        question = request.form['question']
        if vector_store:
            response = generate_response(question, vector_store)
            return jsonify({'response': response})  # Return JSON response
        return jsonify({'error': 'Vector store not initialized.'}), 400
    except Exception as e:
        logging.error(f"Error in ask route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app on localhost with port 8081
    #app.run(host='127.0.0.1', port=8081, debug=True)
    #app.run(debug=True)
    app.run()