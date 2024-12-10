import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import streamlit as st

# Function to load the ratings data
def load_data():
    try:
        ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        print("Ratings data loaded successfully.")
    except FileNotFoundError:
        print("Ratings file not found. Please check the file path.")
        return None
    return ratings

# Function to load movie titles data
def load_movie_titles():
    try:
        movie_titles = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['item_id', 'title'])
        print("Movie titles data loaded successfully.")
    except FileNotFoundError:
        print("Movie titles file not found. Please check the file path.")
        return None
    return movie_titles

# Function to merge ratings data with movie titles data
def merge_data(ratings, movie_titles):
    try:
        all_data = pd.merge(ratings, movie_titles, on='item_id')
        print("Data merged successfully.")
    except KeyError:
        print("Error in merging data. Ensure both datasets have the common 'item_id'.")
        return None
    return all_data

# Function to create a user-movie matrix 
def create_user_movie_matrix(all_data):
    try:
        all_data = all_data.groupby(['user_id', 'title'], as_index=False)['rating'].mean()
        user_movie_ratings = all_data.pivot(index='user_id', columns='title', values='rating').fillna(0)
        print("User-movie matrix created successfully.")
    except KeyError:
        print("Error creating user-movie matrix. Ensure the 'user_id' and 'title' columns exist.")
        return None
    return user_movie_ratings

# Function to calculate the movie similarity using cosine similarity
def calculate_similarity(user_movie_ratings):
    try:
        movie_similarity_matrix = cosine_similarity(user_movie_ratings.T)
        similarity_df = pd.DataFrame(
            movie_similarity_matrix, 
            index=user_movie_ratings.columns, 
            columns=user_movie_ratings.columns
        )
        print("Movie similarity matrix calculated successfully.")
    except ValueError:
        print("Error in calculating similarity. Ensure the user-movie matrix is valid.")
        return None
    return similarity_df

# Function to get movie recommendations based on similarity
def get_recommendations(movie_name, similarity_df, num_recs=5):
    try:
        movie_name = movie_name.lower()
        similarity_df = similarity_df.loc[similarity_df.index.str.lower()]

        if movie_name not in similarity_df.index:
            print(f"Movie '{movie_name}' not found.")
            return None
        similar_movies = similarity_df[movie_name].sort_values(ascending=False)
        recommendations = similar_movies.iloc[1:num_recs+1]
        print(f"Top {num_recs} recommendations for '{movie_name}':")
        print(recommendations)
    except KeyError:
        print("Error retrieving recommendations. Check if the movie exists in the similarity matrix.")
        return None
    return recommendations

# Function to evaluate the system 
def evaluate_system(test_set, user_movie_ratings, similarity_df):
    try:
        test_ratings = test_set.pivot(index='user_id', columns='title', values='rating').fillna(0)
        print(f"Test ratings matrix:\n{test_ratings.head()}")

        predictions = []
        actual = []

        for u in test_ratings.index:
            for m in test_ratings.columns:
                if m in user_movie_ratings.columns:  
                    sim_scores = similarity_df[m]
                    user_scores = user_movie_ratings.loc[u]
                    pred_rating = np.dot(user_scores, sim_scores) / np.sum(sim_scores)

                    predictions.append(pred_rating)
                    actual.append(test_ratings.loc[u, m])
                else:
                    print(f"Warning: Movie '{m}' not found in the user-movie matrix.")
        
        if not predictions or not actual:
            print("Error: No predictions or actual ratings were collected.")
            return None

        threshold = 3.5
        predicted_labels = [1 if pred >= threshold else 0 for pred in predictions]
        actual_labels = [1 if act >= threshold else 0 for act in actual]

        precision = precision_score(actual_labels, predicted_labels)
        recall = recall_score(actual_labels, predicted_labels)
        f1 = f1_score(actual_labels, predicted_labels)

        print(f"System evaluation completed. Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
        return precision, recall, f1
    except ValueError as e:
        print(f"Error in evaluating the system: {e}")
        return None

# Main function to control the workflow
def main():
    ratings = load_data()
    if ratings is None:
        return

    movie_titles = load_movie_titles()
    if movie_titles is None:
        return

    all_data = merge_data(ratings, movie_titles)
    if all_data is None:
        return

    user_movie_ratings = create_user_movie_matrix(all_data)
    if user_movie_ratings is None:
        return

    similarity_df = calculate_similarity(user_movie_ratings)
    if similarity_df is None:
        return

    # Streamlit UI for movie recommendations
    st.title('Movie Recommendation System')

    # User input for movie title
    movie_input = st.text_input('Enter a movie title:', 'Toy Story')

    # Get recommendations
    if movie_input:
        recommendations = get_recommendations(movie_input, similarity_df, num_recs=5)
        if recommendations is not None and not recommendations.empty:
            st.write('Top 5 similar movies:')
            st.write(recommendations)
        else:
            st.write(f"No recommendations found for '{movie_input}'.")

if __name__ == "__main__":
    main()
