"""
Movie Recommendation API — Flask Backend
Mirrors the notebook pipeline exactly:
  Word2Vec avg vectors (weight 0.7) + genre one-hot (weight 0.3)
  Bayesian weighted rating blended in final score (weight 0.2)
  Cosine similarity for ranking

Run:
    pip install flask flask-cors gensim pandas scikit-learn nltk requests numpy
    python app.py
Then open index.html in your browser.
"""

import ast, re, os, requests
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, MinMaxScaler
from nltk.tokenize import RegexpTokenizer

app = Flask(__name__)
CORS(app)

# ── FILE PATHS ── adjust if needed ───────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
METADATA_CSV = os.path.join(BASE_DIR, "movies_metadata.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "movie_words.model")

# ── FIX 1: TMDB API KEY ──────────────────────────────────────────────────────
# os.environ.get(KEY, default) — first arg must be the ENV VARIABLE NAME,
# not the actual key string. Put your real key as the second argument,
# or set the environment variable TMDB_API_KEY before running.
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "e3ca63b9934af81483ad22aed163d74b")

# ── LOAD + PREPROCESS (once at startup) ──────────────────────────────────────
print("Loading Word2Vec model …")

# ── FIX 2: NumPy / gensim version mismatch ───────────────────────────────────
# If you get: ValueError: MT19937 is not a known BitGenerator
# Run this in your terminal FIRST:
#   pip install --upgrade numpy gensim
# The model was saved with a newer NumPy; upgrading fixes the pickle mismatch.
w2v = Word2Vec.load(MODEL_PATH)

print("Loading and preprocessing movie metadata …")
df = pd.read_csv(METADATA_CSV, low_memory=False)
df = df.dropna(subset=["overview", "genres", "title"])
df = df[df["overview"].str.strip() != ""].reset_index(drop=True)

# ── FIX 3: vote_average / vote_count may be missing in some CSV versions ─────
for col in ["vote_average", "vote_count"]:
    if col not in df.columns:
        df[col] = 0.0

df = df[["genres", "title", "overview", "vote_average", "vote_count"]].reset_index(drop=True)

# ── FIX 4: vote_average and vote_count must be numeric ───────────────────────
df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
df["vote_count"]   = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0.0)

def parse_genres(x):
    try:   return [i["name"] for i in ast.literal_eval(x)]
    except: return []

df["genres_list"] = df["genres"].apply(parse_genres)
df = df[df["genres_list"].apply(len) > 0].reset_index(drop=True)

# genre one-hot columns
all_genres = sorted({g for gs in df["genres_list"] for g in gs})
for g in all_genres:
    df[g] = df["genres_list"].apply(lambda lst: int(g in lst))

# tokenise overviews (same as notebook)
tokenizer = RegexpTokenizer(r"\w+")
def tokenize(text):
    text = re.sub(r"\(\d{4}\)", "", str(text))
    return tokenizer.tokenize(text.lower())

movie_words = df["overview"].apply(tokenize).tolist()

# Word2Vec sentence vectors (avg pooling)
vecs = []
for words in movie_words:
    wv = [w2v.wv[w] for w in words if w in w2v.wv]
    vecs.append(np.mean(wv, axis=0) if wv else np.zeros(w2v.vector_size))
avg_np = normalize(np.array(vecs))

# genre matrix
genre_matrix = df[all_genres].to_numpy().astype(float)

# combined feature matrix — exact same weights as notebook
movie_features = np.concatenate([avg_np * 0.7, genre_matrix * 0.3], axis=1)

# Bayesian weighted rating
C = df["vote_average"].mean()
m = df["vote_count"].quantile(0.60)
v, R = df["vote_count"], df["vote_average"]
df["weighted_rating"] = (v / (v + m) * R) + (m / (m + v) * C)
scaler = MinMaxScaler()
df["weighted_rating_norm"] = scaler.fit_transform(df[["weighted_rating"]])

# drop NaN rows
nan_rows = np.unique(np.where(np.isnan(movie_features))[0])
mask = np.ones(len(df), dtype=bool)
mask[nan_rows] = False
df_v   = df[mask].reset_index(drop=True)
feat_v = movie_features[mask]

print(f"Ready — {len(df_v):,} movies indexed")

# ── POSTER CACHE ─────────────────────────────────────────────────────────────
_poster_cache: dict = {}

def get_poster(title: str) -> str:
    if not TMDB_API_KEY:
        return ""
    clean = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    if clean in _poster_cache:
        return _poster_cache[clean]
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"query": clean, "api_key": TMDB_API_KEY},
            timeout=4
        )
        results = r.json().get("results", [])
        path = results[0].get("poster_path", "") if results else ""
        url  = f"https://image.tmdb.org/t/p/w300{path}" if path else ""
    except Exception:
        url = ""
    _poster_cache[clean] = url
    return url

# ── CORE FUNCTION ─────────────────────────────────────────────────────────────
def recommend(title: str, n: int = 10):
    hits = df_v[df_v["title"].str.lower() == title.lower()]
    if hits.empty:
        hits = df_v[df_v["title"].str.lower().str.contains(title.lower(), na=False)]
    if hits.empty:
        return None, []

    idx  = hits.sort_values("vote_count", ascending=False).index[0]
    qvec = feat_v[idx].reshape(1, -1)
    sims = cosine_similarity(qvec, feat_v)[0]
    final = 0.8 * sims + 0.2 * df_v["weighted_rating_norm"].values
    top   = np.argsort(final)[::-1][1: n + 1]

    recs = []
    for i in top:
        row = df_v.iloc[i]
        recs.append({
            "title":  row["title"],
            "genres": row["genres_list"],
            "score":  round(float(final[i]), 4),
            "rating": round(float(row["vote_average"]), 1),
            "poster": get_poster(row["title"]),
        })

    qrow = df_v.iloc[idx]
    return {
        "title":  qrow["title"],
        "genres": qrow["genres_list"],
        "rating": round(float(qrow["vote_average"]), 1),
        "poster": get_poster(qrow["title"]),
    }, recs

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/api/recommend")
def api_recommend():
    title = request.args.get("title", "").strip()
    n     = int(request.args.get("n", 10))
    if not title:
        return jsonify({"error": "title is required"}), 400
    queried, recs = recommend(title, n)
    if queried is None:
        return jsonify({"error": f"'{title}' not found"}), 404
    return jsonify({"query": queried, "recommendations": recs})

@app.route("/api/search")
def api_search():
    q   = request.args.get("q", "").strip().lower()
    lim = int(request.args.get("limit", 10))
    if not q:
        return jsonify([])
    hits = df_v[df_v["title"].str.lower().str.contains(q, na=False)].head(lim)
    return jsonify([{"title": r["title"], "genres": r["genres_list"]}
                    for _, r in hits.iterrows()])

@app.route("/api/popular")
def api_popular():
    n   = int(request.args.get("n", 20))
    top = df_v.nlargest(n, "weighted_rating_norm")
    return jsonify([{
        "title":  r["title"],
        "genres": r["genres_list"],
        "rating": round(float(r["vote_average"]), 1),
        "poster": get_poster(r["title"]),
    } for _, r in top.iterrows()])

@app.route("/")
def index():
    return "<h3>✅ API running — open index.html in your browser</h3>"

if __name__ == "__main__":
    app.run(debug=True, port=5000)