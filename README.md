# 🎬 CineMatch — AI-Powered Movie & Music Discovery Platform

> A full-stack Data Science & Machine Learning project featuring a hybrid movie recommender, mood-based music discovery, and an interactive movie trivia game.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🚀 Features

| Tab              | What it does                                                  |
| ---------------- | ------------------------------------------------------------- |
| 🎬 **Movies**    | Genre + decade based recommendations + "Find Similar" search  |
| 🎵 **Music Hub** | Last.fm powered — similar tracks, by artist, by mood, search  |
| 🎮 **Trivia**    | Movie trivia game with countdown timer, streaks, and confetti |

---

## 🧠 ML Algorithms

### 1. Content-Based Filtering (Notebook 02)

- **TF-IDF Vectorization** on genre + decade features
- **Cosine Similarity** matrix (1682 × 1682)
- Genre overlap score: **100%** across all test queries

### 2. Collaborative Filtering (Notebook 03)

- **SciPy SVD** matrix factorization on 94% sparse user-movie matrix
- Mean-centered ratings to remove user bias
- K tuning from 10 → 100 latent factors
- **RMSE: 0.57 | MAE: 0.42** (best K=100)

### 3. Hybrid Recommender

- Blends CF + CB with tunable **alpha weighting**
- Content-based anchored on movies user rated ≥ 4.0
- `Hybrid Score = α × Collaborative + (1-α) × Content`

### 4. K-Means Clustering (Notebook 04)

- Clusters **89,740 Spotify tracks** by audio features
- Features: energy, valence, danceability, tempo, acousticness, etc.
- Optimal **K=7** selected via Elbow + Silhouette Score (0.20)
- Auto-labeled mood clusters: Happy, Dark, Chill, Party, etc.

---

## 🏗️ Project Structure

```
CineMatch/
├── backend/
│   ├── app.py                  ← FastAPI server (15+ endpoints)
│   ├── recommender/
│   │   ├── content_based.py    ← TF-IDF + Cosine Similarity
│   │   ├── collaborative.py    ← SciPy SVD
│   │   └── spotify_rec.py      ← K-Means music clustering
│   ├── models/                 ← Trained .pkl files (gitignored)
│   └── data/                   ← MovieLens + Spotify datasets
├── frontend/
│   └── index.html              ← Single-page cinematic web app
└── notebooks/
    ├── 01_eda.ipynb            ← Exploratory Data Analysis
    ├── 02_content_based.ipynb  ← TF-IDF recommender
    ├── 03_collaborative.ipynb  ← SVD collaborative filtering
    ├── 04_spotify.ipynb        ← K-Means music clustering
    └── 05_mov_recommender.ipynb ← Alternative implementation (comparison)
```

---

## 📊 Datasets

| Dataset                | Source                                                                           | Size                    |
| ---------------------- | -------------------------------------------------------------------------------- | ----------------------- |
| MovieLens Latest-Small | [GroupLens](https://grouplens.org/datasets/movielens/latest/)                    | 100K ratings, 9K movies |
| Spotify Tracks         | [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) | 89,740 tracks           |

---

## 🔌 APIs Used

| API                                          | Purpose                             | Cost                |
| -------------------------------------------- | ----------------------------------- | ------------------- |
| [OMDB API](https://www.omdbapi.com/)         | Movie posters & plot overviews      | Free                |
| [Last.fm API](https://www.last.fm/api)       | Music recommendations & artist data | Free                |
| [Deezer API](https://developers.deezer.com/) | Artist & album artwork              | Free, no key needed |
| [Open Trivia DB](https://opentdb.com/)       | Movie trivia questions              | Free, no key needed |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/prjena987/CineMatch.git
cd CineMatch
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn pandas scikit-learn scipy requests
```

### 3. Download datasets

- **MovieLens**: Download `ml-latest-small.zip` from [GroupLens](https://grouplens.org/datasets/movielens/latest/) → extract `movies.csv` and `ratings.csv` → place in `backend/data/`
- **Spotify**: Download from [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) → rename to `spotify_tracks.csv` → place in `backend/data/`

### 4. Run the notebooks (in order)

```
notebooks/01_eda.ipynb
notebooks/02_content_based.ipynb
notebooks/03_collaborative.ipynb
notebooks/04_spotify.ipynb
```

### 5. Add API keys

Open `frontend/index.html` and add your OMDB key:

```javascript
const TMDB_KEY = "your_omdb_key_here";
```

Open `backend/app.py` and your Last.fm key is already set.

### 6. Start the backend

```bash
cd backend
python -m uvicorn app:app --reload --port 8000
```

### 7. Open the frontend

Open `frontend/index.html` in your browser or use Live Server in VS Code.

---

## 📈 Model Evaluation

### Collaborative Filtering (SVD)

| K (Latent Factors) | RMSE       | MAE        |
| ------------------ | ---------- | ---------- |
| 10                 | 0.9018     | 0.7071     |
| 50                 | 0.7237     | 0.5519     |
| 75                 | 0.6421     | 0.4801     |
| **100**            | **0.5721** | **0.4186** |

### K-Means Clustering (Music)

| Metric           | Value  |
| ---------------- | ------ |
| Optimal K        | 7      |
| Silhouette Score | 0.20   |
| Tracks clustered | 89,740 |

---

## 🖥️ Tech Stack

**Backend:** Python · FastAPI · Scikit-learn · SciPy · Pandas · NumPy  
**Frontend:** Vanilla JavaScript · HTML5 · CSS3  
**ML:** TF-IDF · Cosine Similarity · SVD · K-Means · PCA  
**APIs:** Last.fm · OMDB · Deezer · Open Trivia DB

---

## 👤 Author

**Pranab Ranjan Jena**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/pranabranjanjena)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/prjena987)
