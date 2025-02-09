from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the ONNX model
model_name_or_path = "myONNXmodels/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name_or_path, backend='onnx')

@app.route('/embed', methods=['GET'])
def get_embedding():
    sentence = request.args.get('sentence')
    if not sentence:
        return jsonify({"error": "Missing 'sentence' parameter"}), 400
    
    # Generate embedding
    embedding = model.encode([sentence])
    
    return jsonify({"embedding": embedding.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
