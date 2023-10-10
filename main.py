import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the dataset (replace <path_to_dataset> with the actual path)
dataset = pd.read_csv('./dataset/music_data.csv')

# Perform one-hot encoding for song titles, artist names, and genres
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(dataset[['title', 'artist', 'genre']])

def recommend_songs(query, top_n=10):
    # Check if the query matches any song title exactly
    song_matches = dataset[dataset['title'] == query]
    
    if len(song_matches) == 0:
        # Check if the query matches any artist name exactly
        artist_matches = dataset[dataset['artist'] == query]
        
        if len(artist_matches) == 0:
            print("No songs found for the given query.")
            return []
        
        # Get the artist name
        artist = artist_matches['artist'].iloc[0]
        
        # Get all songs by the artist
        artist_songs = dataset[dataset['artist'] == artist][['title', 'artist']]
        
        # Display the list of songs by the artist
        print(f"Songs by the artist '{artist}':")
        lst = []
        for index, row in artist_songs.iterrows():
            song = f"{row['title']} - {row['artist']}"
            print(song)
            lst.append(song)
        
        return lst
    else:
        # Get the song name
        song_name = song_matches['title'].iloc[0]
        
        # Get the artist and genre of the song
        artist = song_matches['artist'].iloc[0]
        genre = song_matches['genre'].iloc[0]
        
        # Get the indices of similar songs with the same artist
        artist_similar_songs_indices = [
            i for i in range(len(dataset))
            if dataset['artist'].iloc[i] == artist and i != song_matches.index[0]
        ]
        
        # Get the indices of similar songs with the same genre
        genre_similar_songs_indices = [
            i for i in range(len(dataset))
            if dataset['genre'].iloc[i] == genre and i != song_matches.index[0]
        ]
        
        # Reshape the input features to have a single sample with multiple features
        input_features = encoded_features[song_matches.index[0]].reshape(1, -1)
        
        # Compute the cosine similarity between the input song and all other songs
        song_cosine_sim = cosine_similarity(input_features, encoded_features)
        
        # Sort the similar songs by similarity score, and then by artist and genre
        similar_songs_indices = sorted(artist_similar_songs_indices, key=lambda i: song_cosine_sim[0][i], reverse=True) + sorted(genre_similar_songs_indices, key=lambda i: song_cosine_sim[0][i], reverse=True)
        
        # Remove duplicates while maintaining the order
        similar_songs_indices = list(dict.fromkeys(similar_songs_indices))
        
        # Get the top N similar songs
        top_similar_songs_indices = similar_songs_indices[:top_n]
        
        # Get the song titles and artist names from the dataset
        recommended_songs = [f"{dataset['title'].iloc[i]} - {dataset['artist'].iloc[i]}" for i in top_similar_songs_indices]
        
        # Move the searched song to the front of the list
        recommended_songs.insert(0, f"{song_name} - {artist}")
        
        # Display the list of recommended songs
        print(f"Recommended songs based on '{song_name}':")
        lst = []
        for song in recommended_songs:
            print(song)
            lst.append(song)
        
        return lst
            
@app.post("/search/")
async def searchSong(song: str):
    lst = recommend_songs(song)
    return {"message": json.dumps(lst)}