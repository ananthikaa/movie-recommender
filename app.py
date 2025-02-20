import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Function to extract top 3 actors
def get_top_actors(cast):
    try:
        cast_list = ast.literal_eval(cast)  # Convert string to list of dictionaries
        return [actor["name"] for actor in cast_list[:3]]  # Extract top 3 actors
    except:
        return []

# Apply function to extract actors
df["actors"] = df["cast"].apply(get_top_actors)

# Function to extract director name
def get_director(crew):
    try:
        crew_list = ast.literal_eval(crew)
        for person in crew_list:
            if person["job"] == "Director":
                return person["name"]
        return np.nan
    except:
        return np.nan

# Apply function to extract director
df["director"] = df["crew"].apply(get_director)

# Function to create a combined feature string
def create_soup(row):
    return " ".join(row["actors"]) + " " + (row["director"] if row["director"] is not np.nan else "")

# Apply function to create the soup
df["soup"] = df.apply(create_soup, axis=1)

# Initialize CountVectorizer
vectorizer = CountVectorizer(stop_words="english")

# Transform text into numerical vectors
count_matrix = vectorizer.fit_transform(df["soup"])

# Compute cosine similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Reset index
df = df.reset_index()

# Create a mapping of movie titles to their index
indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return "Movie not found!"

    idx = indices[title]  # Get index of the movie
    sim_scores = list(enumerate(cosine_sim[idx]))  # Compute similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10
    movie_indices = [i[0] for i in sim_scores]  # Get movie indices

    return df["title"].iloc[movie_indices]  # Return similar movie titles

import streamlit as st

st.title("Movie Recommendation System")

# Create a form to capture the input and trigger when Enter is pressed
with st.form(key="movie_form"):
    movie_name = st.text_input("Enter a movie title:")
    submit_button = st.form_submit_button(label="Recommend")  # Triggers on Enter

if submit_button:
    recommendations = get_recommendations(movie_name)

    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)
