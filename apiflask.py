from flask import Flask, request, jsonify
import numpy as np
import pickle
import stanza
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from gensim.models import Word2Vec
import logging, warnings, os, re, string, gc
from tensorflow.keras.models import load_model
import tensorflow as tf

# Silence warnings
logging.basicConfig(level=logging.WARNING)
logging.getLogger('stanza').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # no GPU probing at all
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # quiet logs
os.environ["TF_XLA_FLAGS"] = "--xla_gpu=false"  # don’t spin up XLA GPU backend


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

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

MODEL_PATH = "model_save_ml/ml_model_lstm.h5"
model = load_model(MODEL_PATH, compile=False)

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
    return padded

def keras_predict(x: np.ndarray) -> np.ndarray:
    preds = model.predict(x, verbose=0)
    return preds[0]

def predict_status_with_probabilities(text_raw):
    x = preprocess_input(text_raw)
    probs = keras_predict(x)
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
