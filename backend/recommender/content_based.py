import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
 
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
 
# ─── Load artifacts once at startup ──────────────────────────────────────────
with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
    tfidf = pickle.load(f)
 
with open(os.path.join(MODELS_DIR, 'movies_processed.pkl'), 'rb') as f:
    movies = pickle.load(f)
 
with open(os.path.join(MODELS_DIR, 'similarity_matrix.pkl'), 'rb') as f:
    sim_matrix = pickle.load(f)
 
with open(os.path.join(MODELS_DIR, 'movie_index.pkl'), 'rb') as f:
    movie_index = pickle.load(f)
 
print("✅ Content-based artifacts loaded!")
 
 
def recommend_from_preferences(genres: list, decade: int = None, n: int = 10, min_ratings: int = 20):
    """
    Given a list of genres and optional decade,
    return top N content-based movie recommendations.
    """
    query_parts = []
    if genres:
        genre_str = ' '.join(genres)
        query_parts.append(f"{genre_str} {genre_str}")
    if decade:
        query_parts.append(f"decade_{decade}s")
 
    if not query_parts:
        return []
 
    query_soup = ' '.join(query_parts)
    query_vec  = tfidf.transform([query_soup])
    query_sim  = cosine_similarity(query_vec, tfidf.transform(movies['soup'])).flatten()
 
    result = movies.copy()
    result['similarity_score'] = query_sim
    result = result[result['rating_count'] >= min_ratings]
    result = result.sort_values(['similarity_score', 'rating_mean'], ascending=[False, False]).head(n)
 
    return result[['movie_id', 'title_clean', 'genres', 'year', 'rating_mean', 'rating_count', 'similarity_score']]\
        .round(3).to_dict(orient='records')
 
 
def recommend_similar(title: str, n: int = 10):
    """
    Given a movie title, return top N most similar movies.
    """
    if title not in movie_index:
        return []
 
    idx        = movie_index[title]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
 
    indices = [i[0] for i in sim_scores]
    scores  = [round(i[1], 4) for i in sim_scores]
 
    result = movies.iloc[indices][['movie_id', 'title_clean', 'genres', 'year', 'rating_mean', 'rating_count']].copy()
    result['similarity_score'] = scores
    return result.round(3).to_dict(orient='records')
 
 
def get_all_genres():
    """Return sorted list of all available genres."""
    all_genres = set()
    for g in movies['genres'].dropna():
        for genre in g.split('|'):
            all_genres.add(genre.strip())
    return sorted(list(all_genres))