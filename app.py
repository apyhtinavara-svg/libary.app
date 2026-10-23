import json
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Загружаем каталог
with open("library_catalog.json", encoding="utf-8") as f:
    BOOKS = json.load(f)

# Загружаем embeddings (создаются один раз через create_embeddings.py)
EMBEDDINGS = None
MODEL = None

def load_embeddings():
    global EMBEDDINGS, MODEL
    if os.path.exists("books_embeddings.pkl"):
        with open("books_embeddings.pkl", "rb") as f:
            EMBEDDINGS = pickle.load(f)
        print("✅ Embeddings загружены из файла")
    else:
        print("⚠️ Файл books_embeddings.pkl не найден. Запустите: python create_embeddings.py")

def get_model():
    global MODEL
    if MODEL is None:
        from sentence_transformers import SentenceTransformer
        MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return MODEL

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()
    genre = request.args.get("genre", "").strip()
    limit = int(request.args.get("limit", 20))

    if not query and not genre:
        # Без запроса — возвращаем все (с пагинацией)
        results = BOOKS
        if genre:
            results = [b for b in results if b.get("genre", "") == genre]
        return jsonify(results[:limit])

    # Фильтр по жанру
    pool = BOOKS
    if genre:
        pool = [b for b in pool if b.get("genre", "") == genre]

    if not query:
        return jsonify(pool[:limit])

    # Семантический поиск
    if EMBEDDINGS is not None and len(EMBEDDINGS) == len(BOOKS):
        model = get_model()
        q_vec = model.encode([query])

        if genre:
            indices = [i for i, b in enumerate(BOOKS) if b.get("genre", "") == genre]
            sub_emb = EMBEDDINGS[indices]
            scores = cosine_similarity(q_vec, sub_emb)[0]
            ranked = sorted(zip(indices, scores), key=lambda x: -x[1])
        else:
            scores = cosine_similarity(q_vec, EMBEDDINGS)[0]
            ranked = sorted(enumerate(scores), key=lambda x: -x[1])

        results = []
        for idx, score in ranked[:limit]:
            book = dict(BOOKS[idx])
            book["score"] = round(float(score) * 100)
            results.append(book)
        return jsonify(results)

    else:
        # Fallback: простой текстовый поиск
        q = query.lower()
        results = []
        for book in pool:
            text = f"{book.get('title','')} {book.get('author','')} {book.get('short_description','')}".lower()
            if q in text:
                results.append(book)
        return jsonify(results[:limit])

@app.route("/api/genres")
def genres():
    g = sorted(set(b.get("genre", "") for b in BOOKS if b.get("genre")))
    return jsonify(g)

if __name__ == "__main__":
    load_embeddings()
    app.run(debug=True, port=5000)
