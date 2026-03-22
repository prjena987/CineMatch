import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

DATA_DIR   = 'data/'
MODELS_DIR = 'models/'

print('Loading movies.csv...')
movies = pd.read_csv(DATA_DIR + 'movies.csv')

# Rebuild soup column
def build_soup(row):
    genres_str = row['genres'].replace('|', ' ') if pd.notna(row['genres']) else ''
    decade_str = f"decade_{int(row['year']//10*10)}s" if pd.notna(row['year']) else ''
    return f"{genres_str} {genres_str} {decade_str}"

movies['soup'] = movies.apply(build_soup, axis=1)

# Rebuild TF-IDF + similarity matrix
print('Rebuilding TF-IDF...')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['soup'])
sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_index = pd.Series(movies.index, index=movies['title_clean']).drop_duplicates()

# Save all
print('Saving pkl files...')
with open(MODELS_DIR + 'movies_processed.pkl', 'wb') as f:
    pickle.dump(movies, f)
with open(MODELS_DIR + 'tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open(MODELS_DIR + 'similarity_matrix.pkl', 'wb') as f:
    pickle.dump(sim_matrix, f)
with open(MODELS_DIR + 'movie_index.pkl', 'wb') as f:
    pickle.dump(movie_index, f)

print('✅ All content-based pkls regenerated!')

# Rebuild predicted_ratings for collaborative
print('Rebuilding SVD predicted ratings...')
ratings = pd.read_csv(DATA_DIR + 'ratings.csv')
rating_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
matrix_values = rating_matrix.values.astype(float)

user_ratings_mean = np.true_divide(
    matrix_values.sum(axis=1),
    (matrix_values != 0).sum(axis=1)
)
matrix_demeaned = matrix_values.copy()
for i in range(matrix_demeaned.shape[0]):
    rated_mask = matrix_demeaned[i] != 0
    matrix_demeaned[i][rated_mask] -= user_ratings_mean[i]

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

sparse_matrix = csr_matrix(matrix_demeaned)
U, sigma, Vt = svds(sparse_matrix, k=100)
U = U[:, ::-1]; sigma = sigma[::-1]; Vt = Vt[::-1, :]

predicted = np.dot(np.dot(U, np.diag(sigma)), Vt)
predicted += user_ratings_mean.reshape(-1, 1)
predicted = np.clip(predicted, 1, 5)

predicted_df = pd.DataFrame(predicted, index=rating_matrix.index, columns=rating_matrix.columns)

with open(MODELS_DIR + 'predicted_ratings.pkl', 'wb') as f:
    pickle.dump(predicted_df, f)
with open(MODELS_DIR + 'svd_model.pkl', 'wb') as f:
    pickle.dump({'predicted_df': predicted_df, 'user_ids': rating_matrix.index.tolist(),
                 'movie_ids': rating_matrix.columns.tolist()}, f)

print('✅ All SVD pkls regenerated!')
print('\n✅ Done! Now run: python -m uvicorn app:app --reload --port 8000')