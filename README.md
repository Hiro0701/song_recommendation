# Song recommendation
A simple song recommendation with Spotipy   
Datasets from [Kaggle](https://www.kaggle.com/code/vatsalmavani/music-recommendation-system-using-spotify-dataset/data)
***
I've made a simple song recommendation with [spotipy](https://github.com/spotipy-dev/spotipy)    
K-means was used to classify songs into 5 clusters (upbeat, decent, mood, unique, club)    
Function takes \[song,artist,released date\] and number of recommended songs as inputs
```python
if __name__ == '__main__':
    print_songs()
```
```
Please enter your favorite songs: light switch/charlie puth/2022/levitating/dua lipa/2020
#Format of <title/artist/released date> and takes multiple songs
How many songs do you want to get?: 10
#Default number is 10
It seems like you usually listen to mood!
Here's some songs for you: 
mood :
['Rio De Janeiro Blue', "['Joe Sample', 'Randy Crawford']", '2007']
['The Lions and the Cucumber', "['Vampire Sound Inc.']", '1969']
["I Don't Mean It", "['R. Kelly']", '2000-11-07']
['LOVE SCENARIO', "['iKON']", '2018-01-25']
['The Ride', "['Drake']", '2011-11-15']
['Quiero Decirte', "['Costumbre']", '2001-01-01']
['Firestone', "['Kygo', 'Conrad Sewell']", '2016-05-13']
["That's The Way Love Is", "['The Isley Brothers']", '1967-01-01']
["That's All - Single Version", "['Genesis']", '1983']
["There Ain't No Good Chain Gang (with Waylon Jennings)", "['Johnny Cash', 'Waylon Jennings']", '1978-05-01']
```
