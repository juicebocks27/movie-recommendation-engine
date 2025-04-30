import string
import pickle
import pandas as pd
import ast
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Object for porterStemmer
ps = PorterStemmer()
nltk.download('stopwords')


def get_genres(obj):
    lista = ast.literal_eval(obj)
    l1 = []
    for i in lista:
        l1.append(i['name'])
    return l1


def get_cast(obj):
    a = ast.literal_eval(obj)
    l_ = []
    len_ = len(a)
    for i in range(0, 10):
        if i < len_:
            l_.append(a[i]['name'])
    return l_


def get_crew(obj):
    l1 = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l1.append(i['name'])
            break
    return l1


def read_csv_to_df():
    #  Reading both the csv files
    credit_ = pd.read_csv(r'files/tmdb_5000_credits.csv')
    movies = pd.read_csv(r'files/tmdb_5000_movies.csv')

    # Merging the dataframes
    movies = movies.merge(credit_, on='title')

    movies2 = movies
    movies2.drop(['homepage', 'tagline'], axis=1, inplace=True)
    movies2 = movies2[['movie_id', 'title', 'budget', 'overview', 'popularity', 'release_date', 'revenue', 'runtime',
                       'spoken_languages', 'status', 'vote_average', 'vote_count']]

    #  Extracting important and relevant features
    movies = movies[
        ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'production_companies', 'release_date']]
    movies.dropna(inplace=True)

    # df[df['column_name'] == some_condition]['target_column'] = new_value
    # df.loc[df['column_name'] == some_condition, 'target_column'] = new_value

    #  Applying functions to convert from list to only items.
    movies['genres'] = movies['genres'].apply(get_genres)
    movies['keywords'] = movies['keywords'].apply(get_genres)
    movies['top_cast'] = movies['cast'].apply(get_cast)
    movies['director'] = movies['crew'].apply(get_crew)
    movies['prduction_comp'] = movies['production_companies'].apply(get_genres)

    #  Removing spaces from between the lines
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tcast'] = movies['top_cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tcrew'] = movies['director'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tprduction_comp'] = movies['prduction_comp'].apply(lambda x: [i.replace(" ", "") for i in x])

    # Creating a tags where we have all the words together for analysis
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['tcast'] + movies['tcrew']

    #  Creating new dataframe for the analysis part only.
    new_df = movies[['movie_id', 'title', 'tags', 'genres', 'keywords', 'tcast', 'tcrew', 'tprduction_comp']]

    # new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['genres'] = new_df['genres'].apply(lambda x: " ".join(x))
    # new_df['keywords'] = new_df['keywords'].apply(lambda x: " ".join(x))
    new_df['tcast'] = new_df['tcast'].apply(lambda x: " ".join(x))
    new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: " ".join(x))

    new_df['tcast'] = new_df['tcast'].apply(lambda x: x.lower())
    new_df['genres'] = new_df['genres'].apply(lambda x: x.lower())
    new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: x.lower())

    #  Applying stemming on tags and tags and keywords
    new_df['tags'] = new_df['tags'].apply(stemming_stopwords)
    new_df['keywords'] = new_df['keywords'].apply(stemming_stopwords)

    return movies, new_df, movies2


def stemming_stopwords(li):
    ans = []

    # ps = PorterStemmer()

    for i in li:
        ans.append(ps.stem(i))

    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for w in ans:
        w = w.lower()
        if w not in stop_words:
            filtered_sentence.append(w)

    str_ = ''
    for i in filtered_sentence:
        if len(i) > 2:
            str_ = str_ + i + ' '

    # Removing Punctuations
    punc = string.punctuation
    str_.translate(str_.maketrans('', '', punc))
    return str_
