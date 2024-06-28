import streamlit as st
import pickle
import pandas as pd
import requests
from requests.exceptions import RequestException, HTTPError
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_poster(mvy_id):
    url = f"https://api.themoviedb.org/3/movie/{mvy_id}?api_key=ffcaec3eb22f1cd5cdd8f5c855639b5a&language=en-US"
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Raise an HTTPError for bad responses
    data = response.json()
    poster_path = data.get('poster_path', '')
    if poster_path:
        full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return full_path
    else:
        return None

def recommend(movie):
    try:
        mvy_index = movies[movies['title'] == movie].index[0]
    except IndexError:
        st.error("Movie not found in database.")
        return [], []

    distance = similarity[mvy_index]
    mvy_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_mvys = []
    recommended_mvyPosters = []
    for i in mvy_list:
        mvy_id = movies.iloc[i[0]].movie_id
        try:
            poster = fetch_poster(mvy_id)
        except (RequestException, HTTPError) as e:
            st.error(f"Error fetching poster for movie ID {mvy_id}: {e}")
            poster = None
        if poster:
            recommended_mvyPosters.append(poster)
            recommended_mvys.append(movies.iloc[i[0]].title)
    return recommended_mvys, recommended_mvyPosters

# Load data
try:
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Data file not found: {e}")
    st.stop()

# Streamlit UI

st.title('Your Movie Recommender!')
selected_movie = st.selectbox("Select a movie", movies['title'].values)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    if names and posters:
        num_cols = len(names)
        cols = st.columns(num_cols)

        for i in range(num_cols):
            with cols[i]:
                st.text(names[i])
                st.image(posters[i])
    else:
        st.warning("No recommendations found.")
