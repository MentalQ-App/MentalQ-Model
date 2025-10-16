from flask import Flask, request, jsonify
import numpy as np
import json
# REMOVE this import: from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import stanza
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from gensim.models import Word2Vec
import logging
import warnings
import os
import re
import string
from tensorflow.lite.python.interpreter import Interpreter


# Silence warnings
logging.basicConfig(level=logging.WARNING)
logging.getLogger('stanza').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# Inisialisasi Stanza untuk tokenisasi dan lemmatization
nlp = stanza.Pipeline('id', processors='tokenize,lemma', use_gpu=False)

# --- Remove Keras .h5 loading ---
# model = load_model('model_save_ml/ml_model_lstm.h5')

# Load Label Encoder
with open('data/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load Word2Vec model dan word_index
word2vec_model = Word2Vec.load("model_word2vec/word2vec_model_MentalQ.model")
word_index = {word: i + 1 for i, word in enumerate(word2vec_model.wv.index_to_key)}
embedding_dim = 100
max_sequence_length = 100

# ========= TFLite: init interpreter once =========
TFLITE_PATH = "model_save_ml/ml_model_lstm.tflite"
interpreter = Interpreter(model_path=TFLITE_PATH, num_threads=os.cpu_count() or 1)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# (Usually single input/output for sequence models)
# Confirm expected dtype/shape:
_input_idx = input_details[0]["index"]
_input_dtype = input_details[0]["dtype"]    # often int32 for sequences
_input_shape = input_details[0]["shape"]    # e.g., [1, 100]

_output_idx = output_details[0]["index"]
# ================================================

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Fungsi untuk menghapus stopwords
def remove_stopwords(text):
    factory = StopWordRemoverFactory()
    stop_words = set(factory.get_stop_words())

    manual_stopwords = {"aku", "kamu", "dia", "mereka", "kita", "kami", "mu", "ku", "nya", "itu", "ini", "sini", "situ", "sana", "begitu", "yaitu", "yakni"}
    stop_words.update(manual_stopwords)
    
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Fungsi untuk lemmatization dan tokenisasi menggunakan Stanza
def lemmatize_and_tokenize_text(text):
    doc = nlp(text)
    tokens = []
    lemmatized_text = []
    for sentence in doc.sentences:
        for word in sentence.words:
            tokens.append(word.text)
            lemmatized_text.append(word.lemma)
    return lemmatized_text, tokens

# Fungsi untuk mengonversi token menjadi sequence of integers
def text_to_sequence(tokens, word_index):
    return [word_index[word] for word in tokens if word in word_index]

# Fungsi untuk preprocessing teks
def preprocess_input(text_raw):
    text_raw = clean_text(text_raw)
    text_raw = text_raw.lower()
    text_raw = remove_stopwords(text_raw)
    lemmatized_text, tokenized_text = lemmatize_and_tokenize_text(text_raw)
    sequence = text_to_sequence(tokenized_text, word_index)
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length, padding='post')
    return padded_sequence  # shape (1, max_sequence_length)

# ========= TFLite predict helper =========
def tflite_predict(padded_sequence: np.ndarray) -> np.ndarray:
    """
    Runs the TFLite interpreter and returns probabilities (1D array).
    """
    # Make sure dtype matches the model’s input
    x = padded_sequence.astype(_input_dtype)  # often int32
    # Some exported models expect exact input shape; if needed, resize:
    if list(x.shape) != list(_input_shape):
        interpreter.resize_tensor_input(_input_idx, x.shape, strict=False)
        interpreter.allocate_tensors()

    interpreter.set_tensor(_input_idx, x)
    interpreter.invoke()
    probs = interpreter.get_tensor(_output_idx)  # shape (1, num_classes)
    return probs[0]
# =========================================

# Fungsi untuk melakukan prediksi dengan probabilitas
def predict_status_with_probabilities(text_raw):
    preprocessed_input = preprocess_input(text_raw)
    class_probs = tflite_predict(preprocessed_input)
    predicted_class_idx = int(np.argmax(class_probs, axis=-1))
    predicted_class_prob = float(np.max(class_probs))
    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
    return predicted_label, class_probs

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Invalid input, expected JSON format"}), 400
    
    try:
        data = request.get_json()
        if 'statements' not in data:
            return jsonify({"error": "Missing 'statements' in request"}), 400
        
        statements = data['statements']
        if not isinstance(statements, list):
            return jsonify({"error": "Input data must be a list of statements"}), 400
        
        response = []
        classes = label_encoder.classes_

        for statement in statements:
            predicted_status, class_probabilities = predict_status_with_probabilities(statement)
            confidence_scores = {label: float(prob) for label, prob in zip(classes, class_probabilities)}
            response.append({
                "confidence_scores": confidence_scores,
                "predicted_status": predicted_status,
                "statement": statement
            })
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
