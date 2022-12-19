import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Variables
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='63dad2b096294118b28cf733753fb84b',
                                                           client_secret='3e2e8ba5f61b49b6ab576a4676e3f3ee'))
song_data = pd.read_csv('data.csv')
cluster_dic = {0: 'upbeat', 1: 'decent', 2: 'mood', 3: 'unique', 4: 'club'}

#Data processing
def song_process(song_data):
    song_features = song_data[
        ['valence', 'year', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'popularity',
         'speechiness', 'tempo']]

    return song_features

#Machine Learning
def song_fitting(song):
    pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=5))])
    pipeline.fit(song)
    song['k_mean'] = pipeline.predict(song)

    return song, pipeline

#Spotify search engine with spotipy
def song_search(title, artist, year):
    song = sp.search(q='artist:%s track:%s year:%s' % (artist, title, year), limit=1)

    if song['tracks']['items'] == []:
        return None

    track_id = song['tracks']['items'][0]['id']
    features = sp.audio_features(track_id)[0]

    song_df = pd.DataFrame()

    song_df['name'] = [title]
    song_df['artist'] = [artist]
    song_df['valence'] = features['valence']
    song_df['year'] = [year]
    song_df['acousticness'] = features['acousticness']
    song_df['danceability'] = features['danceability']
    song_df['energy'] = features['energy']
    song_df['instrumentalness'] = features['instrumentalness']
    song_df['loudness'] = features['loudness']
    song_df['popularity'] = song['tracks']['items'][0]['popularity']
    song_df['speechiness'] = features['speechiness']
    song_df['tempo'] = features['tempo']

    return song_df

#Classify song into 5 clusters
def classify_song(song, pipeline):
    return pipeline.predict(song.iloc[:,2:])

#Gives a set of recommended songs
def recommend_song(song_datas, n=10):
    song_df = song_search(song_datas[0][0],song_datas[0][1],song_datas[0][2])

    if len(song_datas) > 1:
        for i in range(len(song_datas)-1):
            temp = song_search(song_datas[i+1][0],song_datas[i+1][1],song_datas[i+1][2])
            song_df = song_df.append(temp, ignore_index=True)

    songs = song_process(song_data)
    (song_features, pipeline) = song_fitting(songs)

    song_classified = classify_song(song_df, pipeline)
    song_classified_list = song_classified.tolist()

    count = []
    for i in range(5):
        count.append(song_classified_list.count(i))

    max_val = max(count)
    cluster_idx = [i for i, v in enumerate(count) if v == max_val]

    recommended_songs = {}
    for i in cluster_idx:
        sample_songs = song_features.loc[song_features['k_mean'] == i].sample(n)
        sample_songs_idx = sample_songs.index.values
        sample_songs = song_data.loc[sample_songs_idx]
        sample_songs = sample_songs[['name', 'artists', 'release_date']]

        cluster_songs = []
        for j in sample_songs_idx:
            cluster_songs.append(sample_songs.loc[j].tolist())

        recommended_songs[i] = cluster_songs

    return recommended_songs

#Display
def print_songs():
    song_datas = input("Please enter your favorite songs: ").split('/')
    n = input("How many songs do you want to get?: ")

    if n == '':
        n = 10
    else:
        n = int(n)

    input_len = len(song_datas)
    if input_len == 3:
        song_datas = [song_datas]
    else:
        temp = []
        for i in range(int(input_len/3)):
            temp.append(song_datas[i*3:i*3+3])
        song_datas = temp

    recommended_songs = recommend_song(song_datas, n)
    clustered = recommended_songs.keys()

    favorite_cluster = []
    for i in clustered:
        favorite_cluster.append(cluster_dic[i])

    print('It seems like you usually listen to %s!' % (', '.join(favorite_cluster)))
    print("Here's some songs for you: ")
    for i in recommended_songs:
        print(cluster_dic[i],':')
        for j in recommended_songs[i]:
            print(j)

if __name__ == '__main__':
    print_songs()