# - Importing the dependencies
import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
from PIL import Image
from io import BytesIO

# - The Movie Database (TMDB) Api Key
API_KEY = 'f142d2d0803cb1986611db989d0052e1'

# - Loading the Data from the csv to Pandas Data Frame
@st.cache_data
def load_movie_data():
    movies = pd.read_csv('movies.csv')

    def concatenate_company_names(row):
        production_companies_json = row['production_companies']
        production_companies = json.loads(production_companies_json)
        company_names = [company['name'] for company in production_companies]
        return ' '.join(company_names)

    movies['production_companies'] = movies.apply(concatenate_company_names, axis=1)
    learning_features = ['genres', 'keywords', 'production_companies', 'tagline', 'cast', 'director']

    for feature in learning_features:
        movies[feature] = movies[feature].fillna('')

    learning_features_combined = movies['genres'] + ' ' + movies['keywords'] + ' ' + movies['production_companies'] + ' ' + movies[
        'tagline'] + ' ' + movies['cast'] + ' ' + movies['director']

    vectorizer = TfidfVectorizer()
    feat_vectors = vectorizer.fit_transform(learning_features_combined)
    cos_similarity = cosine_similarity(feat_vectors)

    return movies, cos_similarity

movies, cos_similarity = load_movie_data()

# - Defining the function to get similar movies
@st.cache_data
def get_similar_movies(movie_name, num_movies=20):
    title_list = movies['title'].tolist()
    movie_matches = difflib.get_close_matches(movie_name, title_list)

    if not movie_matches:
        return []

    closest_match = movie_matches[0]
    closest_match_index = movies[movies.title == closest_match]['index'].values[0]

    similarity_scores = list(enumerate(cos_similarity[closest_match_index]))
    similarity_scores_sorted = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similar_movies = []
    index_number = 1

    for movie in similarity_scores_sorted:
        movies_index = movie[0]
        title_by_index = movies[movies.index == movies_index]['title'].values[0]
        if index_number <= num_movies and title_by_index != movie_name:
            similar_movies.append(title_by_index)
            index_number += 1

    return similar_movies

# - Defining the function to search for movies
def get_movie_poster(movie_name):
    # Searching for movies
    search_url = f'https://api.themoviedb.org/3/search/movie'
    params = {
        'api_key': API_KEY,
        'query': movie_name
    }

    response = requests.get(search_url, params=params)
    data = response.json()

    # - Checking the search results
    if data['results']:
        movie_id = data['results'][0]['id']

        movie_details_url = f'https://api.themoviedb.org/3/movie/{movie_id}'
        params = {
            'api_key': API_KEY
        }

        response = requests.get(movie_details_url, params=params)
        movie_details = response.json()

        # Getting the poster path
        poster_path = movie_details['poster_path']
        poster_url = f'https://image.tmdb.org/t/p/w500/{poster_path}'

        # Displaying the movie poster
        poster_response = requests.get(poster_url)
        img = Image.open(BytesIO(poster_response.content))
        return img

# - Defining the function to get the movie url
def get_movie_url(movie_name):
    # Searching for movies
    search_url = f'https://api.themoviedb.org/3/search/movie'
    params = {
        'api_key': API_KEY,
        'query': movie_name
    }

    response = requests.get(search_url, params=params)
    data = response.json()

    # Checking the search results
    if data['results']:
        movie_id = data['results'][0]['id']

        # Create the TMDB movie URL
        movie_url = f'https://www.themoviedb.org/movie/{movie_id}'
        
        return movie_url


# - Streamlit Web application
st.markdown(
        """
        <style>
        .centered-heading {
            text-align: center;
            padding-bottom: 40px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.markdown("<h1 class='centered-heading'>Movie Recommendation App (Enter a Movie. Get Recommendations!)</h1>", unsafe_allow_html=True)

movie_name = st.text_input("Enter a Movie Name:")
if st.button("Get Recommendations"):
    if movie_name:
        st.subheader(f"We recommend below Movies for You!")
        similar_movies = get_similar_movies(movie_name)

        # - 4 posters per row
        num_posters = len(similar_movies)
        num_columns = 4
        num_rows = (num_posters + num_columns - 1) // num_columns
        columns = st.columns(num_columns)

        for i in range(num_rows):
            for j in range(num_columns):
                idx = i * num_columns + j
                if idx < num_posters:
                    similar_movie = similar_movies[idx]
                    similar_movie_url = get_movie_url(similar_movie)
                    columns[j].image(get_movie_poster(similar_movie), caption=similar_movie, use_column_width=True)
                    st.markdown(f"Visit Website - {similar_movie}: {similar_movie_url}")