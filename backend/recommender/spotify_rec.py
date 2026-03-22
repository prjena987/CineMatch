import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
 
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
 
# ─── Load artifacts once at startup ──────────────────────────────────────────
with open(os.path.join(MODELS_DIR, 'kmeans_spotify.pkl'), 'rb') as f:
    kmeans = pickle.load(f)
 
with open(os.path.join(MODELS_DIR, 'spotify_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
 
with open(os.path.join(MODELS_DIR, 'cluster_labels.pkl'), 'rb') as f:
    cluster_labels = pickle.load(f)
 
tracks_df = pd.read_csv(os.path.join(DATA_DIR, 'spotify_tracks_clustered.csv'))
 
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
 
print("✅ Spotify recommender artifacts loaded!")
 
 
def recommend_by_song(track_name: str, artist: str = None, n: int = 10):
    """
    Given a track name, return N similar songs from the same cluster.
    """
    if artist:
        query = tracks_df[
            (tracks_df['track_name'].str.lower() == track_name.lower()) &
            (tracks_df['artists'].str.lower().str.contains(artist.lower()))
        ]
    else:
        query = tracks_df[tracks_df['track_name'].str.lower() == track_name.lower()]
 
    if len(query) == 0:
        return {'error': f"Track '{track_name}' not found.", 'results': []}
 
    query_row     = query.iloc[0]
    query_cluster = int(query_row['cluster'])
    query_vec     = scaler.transform([query_row[AUDIO_FEATURES].values])
 
    cluster_df   = tracks_df[tracks_df['cluster'] == query_cluster].copy()
    cluster_vecs = scaler.transform(cluster_df[AUDIO_FEATURES].values)
    sims         = cosine_similarity(query_vec, cluster_vecs).flatten()
    cluster_df   = cluster_df.copy()
    cluster_df['similarity'] = sims
 
    cluster_df = cluster_df[cluster_df['track_id'] != query_row['track_id']]
    result     = cluster_df.sort_values('similarity', ascending=False).head(n)
 
    return {
        'query': {
            'track_name'  : query_row['track_name'],
            'artists'     : query_row['artists'],
            'cluster'     : query_cluster,
            'mood_label'  : cluster_labels.get(query_cluster, 'Unknown'),
            'energy'      : round(float(query_row['energy']), 3),
            'valence'     : round(float(query_row['valence']), 3),
            'danceability': round(float(query_row['danceability']), 3),
        },
        'results': result[['track_name', 'artists', 'track_genre',
                            'mood_label', 'similarity', 'energy',
                            'valence', 'danceability']].round(3).to_dict(orient='records')
    }
 
 
def recommend_by_mood(energy: float = 0.7, valence: float = 0.7,
                      danceability: float = 0.7, acousticness: float = 0.2, n: int = 10):
    """
    Given mood sliders, recommend tracks that best match.
    """
    query_features = {
        'danceability'     : danceability,
        'energy'           : energy,
        'loudness'         : -10 + energy * 10,
        'speechiness'      : 0.05,
        'acousticness'     : acousticness,
        'instrumentalness' : 0.01,
        'liveness'         : 0.1,
        'valence'          : valence,
        'tempo'            : 80 + danceability * 80
    }
    query_vec  = scaler.transform([[query_features[f] for f in AUDIO_FEATURES]])
    cluster_id = int(kmeans.predict(query_vec)[0])
 
    cluster_df   = tracks_df[tracks_df['cluster'] == cluster_id].copy()
    cluster_vecs = scaler.transform(cluster_df[AUDIO_FEATURES].values)
    sims         = cosine_similarity(query_vec, cluster_vecs).flatten()
    cluster_df['similarity'] = sims
 
    result = cluster_df.sort_values('similarity', ascending=False).head(n)
    return {
        'cluster'   : cluster_id,
        'mood_label': cluster_labels.get(cluster_id, 'Unknown'),
        'results'   : result[['track_name', 'artists', 'track_genre',
                               'mood_label', 'similarity', 'energy',
                               'valence', 'danceability']].round(3).to_dict(orient='records')
    }
 
 
def get_cluster_summary():
   
    summary = []
    for cluster_id, label in cluster_labels.items():
        count = int((tracks_df['cluster'] == cluster_id).sum())
        center_tracks = tracks_df[tracks_df['cluster'] == cluster_id][AUDIO_FEATURES].mean()
        summary.append({
            'cluster_id'   : cluster_id,
            'mood_label'   : label,
            'track_count'  : count,
            'avg_energy'   : round(float(center_tracks['energy']), 3),
            'avg_valence'  : round(float(center_tracks['valence']), 3),
            'avg_dance'    : round(float(center_tracks['danceability']), 3),
        })
    return summary