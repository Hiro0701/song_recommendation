import json
import joblib
import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

current_path = os.path.dirname(os.path.abspath(__file__))
secret_file_path = os.path.join(current_path, 'secret.json')

with open(secret_file_path, 'r') as secret_file:
    secret_data = json.load(secret_file)
    client_id = secret_data.get("client_id", None)
    client_secret = secret_data.get("client_secret", None)

spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret)
)


def get_song_data(path):
    try:
        song_data = pd.read_csv(path)

        return song_data

    except FileNotFoundError:
        raise ValueError("No file found.")


songs = get_song_data(current_path + '/data.csv')


def retrieve_song_features(song_data):
    song_data_features = song_data[['valence',
                                    'year',
                                    'acousticness',
                                    'danceability',
                                    'energy',
                                    'instrumentalness',
                                    'loudness',
                                    'popularity',
                                    'speechiness',
                                    'tempo']]

    return song_data_features


songs_features = retrieve_song_features(songs)


def train_pipeline(songs_features):
    pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=5))])
    pipeline.fit(songs_features)

    return pipeline


pipeline = train_pipeline(songs_features)


# joblib.dump(pipeline, 'trained_pipeline.pkl')

def label_songs(songs_features, pipeline):
    songs_features['k_mean'] = pipeline.predict(songs_features)

    return songs_features


labeled_songs = label_songs(songs_features, pipeline)


def search_by_artist(artist):
    searched_songs = spotify.search(q='artist=%s' % artist)

    if searched_songs['tracks']['total'] == 0:
        raise ValueError('No song searched.')

    songs_df = []

    for song in searched_songs['tracks']['items']:
        track_id = song['id']
        features = spotify.audio_features(track_id)[0]

        songs_data = {
            'valence': features['valence'],
            'year': song['album']['release_date'],
            'acousticness': features['acousticness'],
            'danceability': features['danceability'],
            'energy': features['energy'],
            'instrumentalness': features['instrumentalness'],
            'loudness': features['loudness'],
            'popularity': song['popularity'],
            'speechiness': features['speechiness'],
            'tempo': features['tempo'],
        }
        songs_df.append(songs_data)

    songs_df = pd.DataFrame(songs_df)
    songs_df['year'] = songs_df['year'].str[:4].astype(int)

    songs_mean = songs_df.mean(axis=0)

    return pd.DataFrame(songs_mean).transpose()


def classify_song(song, pipeline):
    return pipeline.predict(song)


def get_recommended_cluster(artists, pipeline):
    cluster_idx = []

    for artist in artists:
        cluster_idx.append(classify_song(search_by_artist(artist), pipeline))

    count = []
    for i in range(5):
        count.append(cluster_idx.count(i))

    max_val = max(count)
    cluster_idx = [i for i, v in enumerate(count) if v == max_val]

    return cluster_idx[0]


def recommend_songs(pipeline, songs, labeled_songs, n=5):
    artists = input('Type your favorite artists with comma: ')

    try:
        n = int(input('How many songs do you want to get? Default is 5: '))
    except ValueError:
        raise ValueError("It's not a number.")

    if ',' in artists:
        artists = artists.split(',')
    else:
        artists = [artists]

    cluster_idx = get_recommended_cluster(artists, pipeline)

    sample_songs = labeled_songs.loc[labeled_songs['k_mean'] == cluster_idx].sample(n)
    sample_songs_idx = sample_songs.index.values
    sample_songs = songs.loc[sample_songs_idx]
    sample_songs = sample_songs[['name', 'artists', 'release_date']]

    cluster_songs = []
    for j in sample_songs_idx:
        cluster_songs.append(sample_songs.loc[j].tolist())

    return cluster_songs, cluster_idx


cluster_dic = {0: 'upbeat', 1: 'decent', 2: 'mood', 3: 'unique', 4: 'club'}

recommended_songs, cluster_idx = recommend_songs(pipeline, songs, labeled_songs, 5)

print(f'Seems like you like {cluster_dic[cluster_idx]} songs!')
print("Here's some song recommendations for you!")

for song in recommended_songs:
    print("*"*10)
    print(f"Title: {song[0]}")
    print(f"Artist: {song[1][1:-1]}")
    print(f"Released date: {song[2]}")