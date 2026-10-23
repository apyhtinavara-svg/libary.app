"""
Запустить один раз:  python create_embeddings.py
Создаёт books_embeddings.pkl — индекс для быстрого поиска.
"""
import json
import pickle
from sentence_transformers import SentenceTransformer

print("Загружаем каталог...")
with open("library_catalog.json", encoding="utf-8") as f:
    books = json.load(f)

# Текст для каждой книги: название + автор + описание
texts = [
    f"{b.get('title', '')} {b.get('author', '')} {b.get('short_description', '')}"
    for b in books
]

print(f"Загружаем модель (первый раз скачивается ~120 МБ)...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print(f"Создаём embeddings для {len(texts)} книг...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

with open("books_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Готово! Файл books_embeddings.pkl сохранён.")
