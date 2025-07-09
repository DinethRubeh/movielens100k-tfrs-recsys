import pandas as pd
import tensorflow as tf
from typing import Tuple
from src.configuration import config as cf
from src.utils.log_utils import logger
from src.models.ranking_model import RecommendationModel

def load_user_movie_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # load user dataset
    users_df = pd.read_csv(cf.USER_DATA_PATH)
    # cast data type
    users_df["user_id"] = users_df["user_id"].astype(str)
    users_df["gender"] = users_df["gender"].astype(str)
    users_df["occupation"] = users_df["occupation"].astype(str)
    # select only the required columns
    users_df = users_df[['user_id', 'gender', 'occupation']]

    # load movie dataset
    movies_df = pd.read_csv(cf.ITEM_DATA_PATH)
    # cast data type
    movies_df["title"] = movies_df["title"].astype(str)
    movies_df["genres"] = movies_df["title"].astype(str)
    
    movies_df = movies_df[['movie_id', 'title', 'genres', 'year']]

    return users_df, movies_df

def recommend_ranked_candidates_for_user(model, user_id:str, candidate_titles:list) -> list:
    
    # for side feature retrieval & to check validity
    users_df, movies_df = load_user_movie_data()

    # Check if user_id exists
    if user_id not in users_df["user_id"].values:
        raise ValueError(f"user_id {user_id} not found")

    # Get user side features or use defaults
    user_row = users_df[users_df["user_id"] == user_id]
    gender = user_row["gender"].iloc[0] if not user_row.empty and pd.notna(user_row["gender"].iloc[0]) else cf.DEFAULT_GENDER
    occupation = user_row["occupation"].iloc[0] if not user_row.empty and pd.notna(user_row["occupation"].iloc[0]) else cf.DEFAULT_OCCUPATION

    # Prepare movie inputs
    movie_titles = []
    movie_genres = []
    for title in candidate_titles:
        movie_row = movies_df[movies_df["title"] == title]
        genre = (
            movie_row["genres"].iloc[0]
            if not movie_row.empty and pd.notna(movie_row["genres"].iloc[0])
            else cf.DEFAULT_GENRE
        )
        movie_titles.append(title)
        movie_genres.append(genre)

    n = len(candidate_titles)

    # prediction
    predictions = model(
        user_id=tf.constant([user_id] * n),
        gender=tf.constant([gender] * n),
        occupation=tf.constant([occupation] * n),
        movie_title=tf.constant(movie_titles),
        genres=tf.constant(movie_genres)
    )

    scores = tf.squeeze(predictions["output_0"], axis=1).numpy()

    # rank
    ranked_candidates = sorted(zip(candidate_titles, scores), key=lambda x: x[1], reverse=True)
    ranked_candidates = [{"movie_title": title, "score": round(float(score), 4)} for title, score in ranked_candidates] # return a list of dicts
    return ranked_candidates

def load_model():
    return tf.saved_model.load(cf.MODEL_PATH).signatures["serving_default"]

def get_recommendations(user_id:str, candidate_list:list) -> list:
    # load the model
    tfrs_model = load_model()
    # get recommendations
    ranked_candidates = recommend_ranked_candidates_for_user(tfrs_model, user_id, candidate_list)
    return ranked_candidates

if __name__ == "__main__":
    user_id = "52"
    candidate_movies = ["Toy Story (1995)", "GoldenEye (1995)", "Four Rooms (1995)"]
    recommendations = get_recommendations(user_id, candidate_movies)
    logger.info(f"inference: {recommendations}")