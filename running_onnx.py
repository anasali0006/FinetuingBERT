from sentence_transformers import SentenceTransformer

model_name_or_path = "myONNXmodels/all-MiniLM-L6-v2"

model = SentenceTransformer(model_name_or_path = model_name_or_path, backend='onnx')

sentences = ["This is an example sentence"]
embedding = model.encode(sentences)
print(embedding)