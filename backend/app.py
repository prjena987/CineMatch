"""
CineMatch — FastAPI Backend
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests, os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'recommender'))

from content_based import recommend_from_preferences, recommend_similar, get_all_genres
from collaborative import recommend_for_user, hybrid_recommend, get_user_history, get_valid_user_ids
from spotify_rec   import recommend_by_song, recommend_by_mood, get_cluster_summary

# ─── Config ───────────────────────────────────────────────────────────────────
LASTFM_KEY  = '0ed7b3937eb9731556519f8de6e52fa0'
LASTFM_BASE = 'http://ws.audioscrobbler.com/2.0/'

app = FastAPI(title="CineMatch API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Request Models ───────────────────────────────────────────────────────────
class MoviePreferences(BaseModel):
    genres : List[str]
    decade : Optional[int] = None
    n      : Optional[int] = 10

class HybridRequest(BaseModel):
    user_id : int
    genres  : Optional[List[str]] = None
    decade  : Optional[int]       = None
    n       : Optional[int]       = 10
    alpha   : Optional[float]     = 0.6

class MoodRequest(BaseModel):
    energy       : float = 0.7
    valence      : float = 0.7
    danceability : float = 0.7
    acousticness : float = 0.2
    n            : int   = 10

# ─── Helpers ──────────────────────────────────────────────────────────────────
def lastfm(params: dict):
    """Make a Last.fm API call and return JSON."""
    params.update({'api_key': LASTFM_KEY, 'format': 'json'})
    r = requests.get(LASTFM_BASE, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def extract_image(images, size='large'):
    """Extract image URL from Last.fm image array."""
    if not images:
        return None
    size_map = {'small': 0, 'medium': 1, 'large': 2, 'extralarge': 3, 'mega': 4}
    idx = size_map.get(size, 2)
    try:
        url = images[min(idx, len(images)-1)]['#text']
        return url if url else None
    except:
        return None
def get_deezer_artist(artist_name: str):
    """Fetch artist image from Deezer (no API key needed)."""
    try:
        r = requests.get(
            f"https://api.deezer.com/search/artist?q={requests.utils.quote(artist_name)}&limit=1",
            timeout=8
        )
        data = r.json()
        artists = data.get('data', [])
        if artists:
            return {
                'image': artists[0].get('picture_xl') or artists[0].get('picture_big'),
                'name' : artists[0].get('name')
            }
    except:
        pass
    return None

def get_deezer_track(track: str, artist: str):
    """Fetch album art from Deezer (no API key needed)."""
    try:
        q = requests.utils.quote(f"{track} {artist}")
        r = requests.get(f"https://api.deezer.com/search?q={q}&limit=1", timeout=8)
        data = r.json()
        tracks = data.get('data', [])
        if tracks:
            album = tracks[0].get('album', {})
            return album.get('cover_xl') or album.get('cover_big')
    except:
        pass
    return None


def clean_track(t: dict, rank: int = None) -> dict:
    """Normalise a Last.fm track object into a consistent shape."""
    artist = t.get('artist', {})
    artist_name = artist.get('name', '') if isinstance(artist, dict) else str(artist)
    images = t.get('image', [])
    return {
        'rank'       : rank,
        'name'       : t.get('name', ''),
        'artist'     : artist_name,
        'playcount'  : t.get('playcount', t.get('match', '')),
        'url'        : t.get('url', ''),
        'image_sm'   : extract_image(images, 'medium'),
        'image_lg'   : extract_image(images, 'extralarge') or extract_image(images, 'large'),
        'duration'   : t.get('duration', ''),
        'listeners'  : t.get('listeners', ''),
    }

# ══════════════════════════════════════════════════════════════════════════════
# ROOT
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"message": "🎬 CineMatch API is running!", "version": "1.0.0"}

# ══════════════════════════════════════════════════════════════════════════════
# 🎬 MOVIES
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/movies/recommend")
def movies_recommend(prefs: MoviePreferences):
    if not prefs.genres:
        raise HTTPException(400, "At least one genre required.")
    results = recommend_from_preferences(genres=prefs.genres, decade=prefs.decade, n=prefs.n)
    return {"query": {"genres": prefs.genres, "decade": prefs.decade}, "count": len(results), "results": results}

@app.get("/movies/similar/{title}")
def movies_similar(title: str, n: int = Query(default=10, ge=1, le=20)):
    results = recommend_similar(title=title, n=n)
    if not results:
        raise HTTPException(404, f"Movie '{title}' not found.")
    return {"query": title, "count": len(results), "results": results}

@app.get("/movies/genres")
def movies_genres():
    return {"genres": get_all_genres()}

@app.get("/movies/toprated")
def movies_toprated(n: int = Query(default=20)):
    """Returns top rated movies for the hero carousel."""
    import pandas as pd
    movies = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'movies.csv'))
    top = (
        movies[movies['rating_count'] >= 50]
        .sort_values('rating_mean', ascending=False)
        .head(n)
    )
    return {"results": top[['movie_id','title_clean','genres','year','rating_mean','rating_count']]
            .fillna(0).to_dict(orient='records')}
# ══════════════════════════════════════════════════════════════════════════════
# ❤️  FOR YOU
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/foryou/hybrid")
def foryou_hybrid(req: HybridRequest):
    results = hybrid_recommend(user_id=req.user_id, genres=req.genres, decade=req.decade, n=req.n, alpha=req.alpha)
    if not results:
        raise HTTPException(404, f"User {req.user_id} not found.")
    return {"user_id": req.user_id, "alpha": req.alpha, "count": len(results), "results": results}

@app.get("/foryou/history/{user_id}")
def foryou_history(user_id: int, n: int = Query(default=10)):
    history = get_user_history(user_id=user_id, n=n)
    if not history:
        raise HTTPException(404, f"No history for user {user_id}.")
    return {"user_id": user_id, "count": len(history), "history": history}

# ══════════════════════════════════════════════════════════════════════════════
# 🎵 LAST.FM MUSIC
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/music/similar")
def music_similar(
    track : str = Query(..., description="Track name"),
    artist: str = Query(..., description="Artist name"),
    n     : int = Query(default=10, ge=1, le=20)
):
    try:
        data = lastfm({'method': 'track.getSimilar', 'track': track, 'artist': artist, 'limit': n, 'autocorrect': 1})
        raw  = data.get('similartracks', {}).get('track', [])
        if not raw:
            raise HTTPException(404, f"No similar tracks found for '{track}' by {artist}.")
        tracks = [clean_track(t, i+1) for i, t in enumerate(raw[:n])]

        # Enrich with Deezer album art for the hero banner
        deezer_img = get_deezer_track(track, artist)
        if deezer_img:
            for t in tracks:
                t['image_lg'] = deezer_img
                break  # only enrich first track for hero

        return {"query": {"track": track, "artist": artist}, "count": len(tracks), "results": tracks}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Last.fm error: {str(e)}")

@app.get("/music/artist/top")
def music_artist_top(
    artist: str = Query(..., description="Artist name"),
    n     : int = Query(default=10, ge=1, le=20)
):
    """Top tracks for a given artist."""
    try:
        data = lastfm({'method': 'artist.getTopTracks', 'artist': artist, 'limit': n, 'autocorrect': 1})
        raw  = data.get('toptracks', {}).get('track', [])
        if not raw:
            raise HTTPException(404, f"Artist '{artist}' not found.")
        tracks = [clean_track(t, i+1) for i, t in enumerate(raw[:n])]

        # Get artist info from Last.fm
        artist_data = lastfm({'method': 'artist.getInfo', 'artist': artist, 'autocorrect': 1})
        artist_info = artist_data.get('artist', {})
        bio_content = artist_info.get('bio', {}).get('summary', '')
        import re
        bio_clean = re.sub(r'<[^>]+>', '', bio_content).split('Read more')[0].strip()

        # Try Deezer for better image — Last.fm images are usually empty
        image = extract_image(artist_info.get('image', []), 'extralarge')
        deezer = get_deezer_artist(artist)
        if deezer and deezer.get('image'):
            image = deezer['image']

        return {
            "artist"   : artist,
            "image"    : image,
            "listeners": artist_info.get('stats', {}).get('listeners', ''),
            "playcount": artist_info.get('stats', {}).get('playcount', ''),
            "bio"      : bio_clean[:300] + '…' if len(bio_clean) > 300 else bio_clean,
            "count"    : len(tracks),
            "results"  : tracks
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Last.fm error: {str(e)}")
@app.get("/music/tag")
def music_by_tag(
    tag: str = Query(..., description="Mood/genre tag e.g. 'chill', 'sad', 'hiphop'"),
    n  : int = Query(default=10, ge=1, le=20)
):
    """
    Get top tracks for a mood/genre tag.
    Good tags: chill, sad, happy, party, workout, study, jazz, classical, hiphop, rock
    """
    try:
        data = lastfm({'method': 'tag.getTopTracks', 'tag': tag, 'limit': n})
        raw  = data.get('tracks', {}).get('track', [])
        if not raw:
            raise HTTPException(404, f"No tracks found for tag '{tag}'.")
        tracks = [clean_track(t, i+1) for i, t in enumerate(raw[:n])]
        return {"tag": tag, "count": len(tracks), "results": tracks}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Last.fm error: {str(e)}")


@app.get("/music/track/info")
def music_track_info(
    track : str = Query(...),
    artist: str = Query(...)
):
    """Full info for a specific track including album art and tags."""
    try:
        data  = lastfm({'method': 'track.getInfo', 'track': track, 'artist': artist, 'autocorrect': 1})
        t     = data.get('track', {})
        album = t.get('album', {})
        tags  = [tag['name'] for tag in t.get('toptags', {}).get('tag', [])[:5]]
        import re
        wiki_raw   = t.get('wiki', {}).get('summary', '')
        wiki_clean = re.sub(r'<[^>]+>', '', wiki_raw).split('Read more')[0].strip()
        return {
            'name'      : t.get('name', ''),
            'artist'    : t.get('artist', {}).get('name', ''),
            'album'     : album.get('title', ''),
            'image_lg'  : extract_image(album.get('image', []), 'extralarge'),
            'image_sm'  : extract_image(album.get('image', []), 'medium'),
            'duration_ms': int(t.get('duration', 0)),
            'listeners' : t.get('listeners', ''),
            'playcount' : t.get('playcount', ''),
            'tags'      : tags,
            'wiki'      : wiki_clean[:400] + '…' if len(wiki_clean) > 400 else wiki_clean,
            'url'       : t.get('url', '')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Last.fm error: {str(e)}")


@app.get("/music/search")
def music_search(
    q: str = Query(..., description="Search query"),
    n: int = Query(default=10, ge=1, le=20)
):
    """Search for tracks by name."""
    try:
        data = lastfm({'method': 'track.search', 'track': q, 'limit': n})
        raw  = data.get('results', {}).get('trackmatches', {}).get('track', [])
        if not raw:
            raise HTTPException(404, f"No results for '{q}'.")
        tracks = [clean_track(t, i+1) for i, t in enumerate(raw[:n])]
        return {"query": q, "count": len(tracks), "results": tracks}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Last.fm error: {str(e)}")


@app.get("/music/moods")
def music_moods():
    """Returns available mood tags with icons."""
    return {"moods": [
        {"tag": "chill",      "label": "Chill",       "icon": "☕", "color": "#58a6ff"},
        {"tag": "sad",        "label": "Sad",         "icon": "😢", "color": "#9b8fff"},
        {"tag": "happy",      "label": "Happy",       "icon": "😊", "color": "#f5c518"},
        {"tag": "party",      "label": "Party",       "icon": "🎉", "color": "#e50914"},
        {"tag": "workout",    "label": "Workout",     "icon": "💪", "color": "#3fcf8e"},
        {"tag": "study",      "label": "Study",       "icon": "📚", "color": "#58a6ff"},
        {"tag": "jazz",       "label": "Jazz",        "icon": "🎷", "color": "#e8840a"},
        {"tag": "classical",  "label": "Classical",   "icon": "🎼", "color": "#c0c0ff"},
        {"tag": "hip-hop",    "label": "Hip-Hop",     "icon": "🎤", "color": "#ff6b35"},
        {"tag": "rock",       "label": "Rock",        "icon": "🤘", "color": "#e85a5a"},
        {"tag": "electronic", "label": "Electronic",  "icon": "🎧", "color": "#3fcf8e"},
        {"tag": "rnb",        "label": "R&B",         "icon": "🎵", "color": "#f5c518"},
    ]}

# ══════════════════════════════════════════════════════════════════════════════
# 🎮 TRIVIA
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/trivia")
def get_trivia(
    amount    : int = Query(default=10, ge=5, le=20),
    difficulty: str = Query(default="medium", pattern="^(easy|medium|hard)$"),
    category  : int = Query(default=11)
):
    url = f"https://opentdb.com/api.php?amount={amount}&category={category}&difficulty={difficulty}&type=multiple"
    try:
        data = requests.get(url, timeout=10).json()
        if data.get('response_code') != 0:
            raise HTTPException(503, "Trivia API returned no results.")
        import random
        questions = []
        for q in data['results']:
            opts = q['incorrect_answers'] + [q['correct_answer']]
            random.shuffle(opts)
            questions.append({
                'question'      : q['question'],
                'correct_answer': q['correct_answer'],
                'options'       : opts,
                'difficulty'    : q['difficulty'],
                'category'      : q['category']
            })
        return {"count": len(questions), "difficulty": difficulty, "questions": questions}
    except requests.exceptions.RequestException:
        raise HTTPException(503, "Could not reach Trivia API.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
