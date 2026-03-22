#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt gdown

echo "Downloading MovieLens dataset..."
wget -q https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip -q ml-latest-small.zip
cp ml-latest-small/movies.csv data/movies.csv
cp ml-latest-small/ratings.csv data/ratings.csv

echo "Downloading Spotify dataset from Google Drive..."
gdown --id 15ZUFCuJIVxPaDl3SZ0Cugf4UKdSrnYU- -O data/spotify_tracks.csv

echo "Building ML models..."
python fix_pkls.py

echo "Build complete!"