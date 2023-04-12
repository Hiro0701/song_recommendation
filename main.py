import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class spotify:
    def __init__(self, client_id, client_secret):
        self.spotify_api = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret)
        )
        self.cluster_dic = {0: 'upbeat', 1: 'decent', 2: 'mood', 3: 'unique', 4: 'club'}

    # Get song data
    def get_song_data(self, path):
        try:
            song_data = pd.read_csv(path)

            return song_data

        except FileNotFoundError:
            raise ValueError("No file found.")

    # Get song data with only useful columns
    def retrieve_song_feature(self, song_data):
        song_data_feature = song_data[['valence',
                                       'year',
                                       'acousticness',
                                       'danceability',
                                       'energy',
                                       'instrumentalness',
                                       'loudness',
                                       'popularity',
                                       'speechiness',
                                       'tempo']]

        return song_data_feature

    # Machine learning
    def train_pipeline(self, song_data):
        pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=5))])
        pipeline.fit(song_data)

        return pipeline

    # Predict songs' classification and label them
    def label_songs(self, song_data, pipeline):
        song_data['k_mean'] = pipeline.predict(song_data)

        return song_data

    # Return dataframe of average info of searched songs by artist
    def search_by_artist(self, artist):
        searched_songs = self.spotify_api.search(q='artist=%s' % artist)

        # Return None if there's no song searched
        if searched_songs['tracks']['total'] == 0:
            raise ValueError('No song searched.')

        songs_df = []

        # Get each song's data
        for song in searched_songs['tracks']['items']:
            track_id = song['id']
            features = self.spotify_api.audio_features(track_id)[0]

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

        # Return songs' mean values
        song_mean = songs_df.mean(axis=0)

        return pd.DataFrame(song_mean).transpose()

    # Classify given song with respect to given pipeline
    def classify_song(self, song, pipeline):

        return pipeline.predict(song)

    # Get recommended cluster index
    def get_recommended_cluster(self, artists, pipeline):
        cluster_idx = []

        for artist in artists:
            cluster_idx.append(pipeline.predict(self.search_by_artist(artist)))

        count = []
        for i in range(5):
            count.append(cluster_idx.count(i))

        max_val = max(count)
        cluster_idx = [i for i, v in enumerate(count) if v == max_val]

        return cluster_idx

    # Recommend n songs by cluster index
    def recommend_songs(self, cluster_idx, song_data, song_labeled, n):
        recommended_songs = {}

        for i in cluster_idx:
            sample_songs = song_labeled.loc[song_labeled['k_mean'] == i].sample(n)
            sample_songs_idx = sample_songs.index.values
            sample_songs = song_data.loc[sample_songs_idx]
            sample_songs = sample_songs[['name', 'artists', 'release_date']]

            cluster_songs = []
            for j in sample_songs_idx:
                cluster_songs.append(sample_songs.loc[j].tolist())

            recommended_songs[i] = cluster_songs

        return recommended_songs

    # Final product
    def recommend(self, path, artist, n=10):
        song_data = self.get_song_data(path)
        song_data_feature = self.retrieve_song_feature(song_data)
        pipeline = self.train_pipeline(song_data_feature)
        song_labeled = self.label_songs(song_data_feature, pipeline)
        song = self.search_by_artist(artist)
        cluster_idx = self.get_recommended_cluster([artist], pipeline)
        recommended_songs = self.recommend_songs(cluster_idx, song_data, song_labeled, n)

        return recommended_songs
