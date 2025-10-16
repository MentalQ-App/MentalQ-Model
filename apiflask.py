from flask import Flask, request, jsonify
import numpy as np
import pickle
import stanza
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from gensim.models import Word2Vec
import logging, warnings, os, re, string, gc

# === Replace TF-Lite import: use lightweight runtime ===
try:
    from tflite_runtime.interpreter import Interpreter   # pip install tflite-runtime
except ImportError:
    # fallback if only TF is available (heavier). Prefer installing tflite-runtime.
    from tensorflow.lite.python.interpreter import Interpreter

# Silence warnings
logging.basicConfig(level=logging.WARNING)
logging.getLogger('stanza').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Stanza once
nlp = stanza.Pipeline('id', processors='tokenize,lemma', use_gpu=False, tokenize_no_ssplit=True)

# Load Label Encoder
with open('data/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Build word_index, then free the big Gensim model
w2v = Word2Vec.load("model_word2vec/word2vec_model_MentalQ.model")
word_index = {w: i + 1 for i, w in enumerate(w2v.wv.index_to_key)}
del w2v
gc.collect()

embedding_dim = 100
max_sequence_length = 100

# === TFLite: init interpreter once ===
TFLITE_PATH = "model_save_ml/ml_model_lstm.tflite"
interpreter = Interpreter(model_path=TFLITE_PATH, num_threads=min(2, (os.cpu_count() or 1)))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_input_idx = input_details[0]["index"]
_input_dtype = input_details[0]["dtype"]      # typically int32
_input_shape = tuple(input_details[0]["shape"])  # e.g. (1, 100)
_output_idx = output_details[0]["index"]

# === Precompute stopwords once ===
_stop_factory = StopWordRemoverFactory()
_stop_words = set(_stop_factory.get_stop_words())
_stop_words.update({"aku","kamu","dia","mereka","kita","kami","mu","ku","nya","itu","ini",
                    "sini","situ","sana","begitu","yaitu","yakni"})

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
    return text.strip()

def remove_stopwords(text):
    words = [w for w in text.split() if w not in _stop_words]
    return ' '.join(words)

def lemmatize_and_tokenize_text(text):
    doc = nlp(text)
    tokens, lemmas = [], []
    for sent in doc.sentences:
        for w in sent.words:
            tokens.append(w.text)
            lemmas.append(w.lemma)
    return lemmas, tokens

def text_to_sequence(tokens, word_index):
    return [word_index[w] for w in tokens if w in word_index]

# === lightweight pad_sequences ===
def pad_sequences_np(seqs, maxlen, padding='post', value=0):
    # seqs: list of 1D lists
    arr = np.full((len(seqs), maxlen), value, dtype=np.int32)
    for i, s in enumerate(seqs):
        s = np.asarray(s, dtype=np.int32)
        if len(s) >= maxlen:
            arr[i] = s[:maxlen]
        else:
            if padding == 'post':
                arr[i, :len(s)] = s
            else:
                arr[i, -len(s):] = s
    return arr

def preprocess_input(text_raw):
    t = clean_text(text_raw.lower())
    t = remove_stopwords(t)
    lemmas, tokens = lemmatize_and_tokenize_text(t)
    seq = text_to_sequence(tokens, word_index)
    padded = pad_sequences_np([seq], maxlen=max_sequence_length, padding='post')
    # Ensure exact expected shape to avoid resize/allocate calls
    if _input_shape == (1, max_sequence_length) and padded.shape != _input_shape:
        padded = padded.reshape(_input_shape)
    return padded

def tflite_predict(x: np.ndarray) -> np.ndarray:
    x = x.astype(_input_dtype, copy=False)
    # Avoid resizing unless absolutely necessary
    if tuple(x.shape) != _input_shape:
        interpreter.resize_tensor_input(_input_idx, x.shape, strict=False)
        interpreter.allocate_tensors()
    interpreter.set_tensor(_input_idx, x)
    interpreter.invoke()
    out = interpreter.get_tensor(_output_idx)
    return out[0]

def predict_status_with_probabilities(text_raw):
    x = preprocess_input(text_raw)
    probs = tflite_predict(x)
    idx = int(np.argmax(probs, axis=-1))
    label = label_encoder.inverse_transform([idx])[0]
    return label, probs

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Invalid input, expected JSON format"}), 400
    data = request.get_json()
    if 'statements' not in data or not isinstance(data['statements'], list):
        return jsonify({"error": "Input data must be a list of statements"}), 400

    classes = label_encoder.classes_
    resp = []
    for s in data['statements']:
        y, p = predict_status_with_probabilities(s)
        resp.append({
            "confidence_scores": {label: float(prob) for label, prob in zip(classes, p)},
            "predicted_status": y,
            "statement": s
        })
    return jsonify(resp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    # Production-safe: no debug, no reloader (prevents duplicate process)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
