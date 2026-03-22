import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')

# ─── Load artifacts once at startup ──────────────────────────────────────────
with open(os.path.join(MODELS_DIR, 'predicted_ratings.pkl'), 'rb') as f:
    predicted_df = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
    tfidf = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'movies_processed.pkl'), 'rb') as f:
    movies_processed = pickle.load(f)

movies  = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))
ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))

# Build rating_matrix index for unseen lookup
rating_matrix = ratings.pivot_table(
    index='user_id', columns='movie_id', values='rating'
).fillna(0)

print("✅ Collaborative filtering artifacts loaded!")


def get_unseen_movies(user_id: int):
    """Returns movie_ids the user has NOT rated."""
    if user_id not in rating_matrix.index:
        return []
    user_row = rating_matrix.loc[user_id]
    return user_row[user_row == 0].index.tolist()


def recommend_for_user(user_id: int, n: int = 10):
    """
    Given a user_id, return top N recommendations
    based on SVD predicted ratings for unseen movies.
    """
    if user_id not in predicted_df.index:
        return []

    unseen_ids   = get_unseen_movies(user_id)
    if not unseen_ids:
        return []

    user_preds   = predicted_df.loc[user_id, unseen_ids]
    top_ids      = user_preds.sort_values(ascending=False).head(n).index.tolist()
    top_scores   = user_preds.sort_values(ascending=False).head(n).values

    results = []
    for mid, score in zip(top_ids, top_scores):
        info = movies[movies['movie_id'] == mid]
        if len(info) > 0:
            r = info.iloc[0]
            results.append({
                'movie_id'         : int(mid),
                'title_clean'      : r['title_clean'],
                'genres'           : r['genres'],
                'year'             : float(r['year']) if not pd.isna(r['year']) else None,
                'predicted_rating' : round(float(score), 3),
                'avg_rating'       : round(float(r['rating_mean']), 3),
                'num_ratings'      : int(r['rating_count'])
            })
    return results


def hybrid_recommend(user_id: int, genres: list = None, decade: int = None,
                     n: int = 10, alpha: float = 0.6):
    """
    Hybrid recommender: α × collaborative + (1-α) × content-based.
    """
    unseen_ids = get_unseen_movies(user_id)
    if not unseen_ids:
        return []

    unseen_df  = movies[movies['movie_id'].isin(unseen_ids)].copy()
    scaler     = MinMaxScaler()

    # Collaborative scores
    collab_preds              = predicted_df.loc[user_id, unseen_df['movie_id'].values]
    unseen_df['collab_score'] = unseen_df['movie_id'].map(collab_preds.to_dict())
    unseen_df['collab_norm']  = scaler.fit_transform(unseen_df[['collab_score']])

    # Content-based scores
    if genres or decade:
        parts = []
        if genres:
            parts.append(' '.join(genres) + ' ' + ' '.join(genres))
        if decade:
            parts.append(f'decade_{decade}s')
        q_vec  = tfidf.transform([' '.join(parts)])
        sims   = cosine_similarity(q_vec, tfidf.transform(movies_processed['soup'])).flatten()
        mid_to = dict(zip(movies_processed['movie_id'], sims))
        unseen_df['content_score'] = unseen_df['movie_id'].map(mid_to).fillna(0)
    else:
        unseen_df['content_score'] = 0.5

    unseen_df['content_norm']  = scaler.fit_transform(unseen_df[['content_score']])
    unseen_df['hybrid_score']  = (
        alpha * unseen_df['collab_norm'] + (1 - alpha) * unseen_df['content_norm']
    )

    result = unseen_df.sort_values('hybrid_score', ascending=False).head(n)
    return result[['movie_id', 'title_clean', 'genres', 'year',
                   'collab_score', 'content_score', 'hybrid_score']]\
        .round(3).to_dict(orient='records')


def get_user_history(user_id: int, n: int = 10):
    """Returns movies the user has already rated (top rated first)."""
    user_ratings = ratings[ratings['user_id'] == user_id].merge(
        movies[['movie_id', 'title_clean', 'genres', 'year']], on='movie_id'
    ).sort_values('rating', ascending=False)
    return user_ratings[['movie_id', 'title_clean', 'genres', 'year', 'rating']]\
        .head(n).to_dict(orient='records')


def get_valid_user_ids():
    """Returns list of valid user IDs in the dataset."""
    return sorted(ratings['user_id'].unique().tolist())