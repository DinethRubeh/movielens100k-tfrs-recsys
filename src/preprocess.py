import pandas as pd
from pathlib import Path
from configuration import config as cf
from utils.log_utils import logger

def preprocess_movies() -> pd.DataFrame:
    # Movies: u.item file (pipe-separated)
    genre_cols = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols

    movies = pd.read_csv(
        f'{cf.RAW_DATA_PATH}/u.item', sep='|', encoding='latin-1', header=None, 
        names=movie_cols)

    # genres as a string
    movies["genres"] = movies[genre_cols].apply(
        lambda row: "|".join([genre for genre, val in row.items() if val == 1]), axis=1)
    # extract release year
    movies["year"] = movies["release_date"].str.extract(r"(\d{4})").fillna("unknown")

    # create directory if not exists
    output_dir = Path(cf.PREPROCESSED_DATA_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # save as a csv
    movies.to_csv(f'{cf.PREPROCESSED_DATA_PATH}/movies.csv', index=False)
    logger.info("Raw movie pre-processing finished.")

    return movies

def preprocess_users() -> pd.DataFrame:
    # Users: u.user file (pipe-separated)
    users = pd.read_csv(
        f'{cf.RAW_DATA_PATH}/u.user', sep='|', header=None, 
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

    # save as a csv
    users.to_csv(f'{cf.PREPROCESSED_DATA_PATH}/users.csv', index=False)
    logger.info("Raw user pre-processing finished.")

    return users

def preprocess_ratings(users:pd.DataFrame, movies:pd.DataFrame) -> None:
    # Ratings: u.data file (tab-separated)
    ratings = pd.read_csv(
        f'{cf.RAW_DATA_PATH}/u.data', sep='\t', header=None, 
        names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # join user features to the ratings
    ratings = ratings.merge(
        users[['user_id', 'age', 'gender', 'occupation']], 
        on="user_id", how="left")

    # join movie features to the ratings
    ratings = ratings.merge(
        movies[["movie_id", "title", "genres", "year"]],
        on="movie_id", how="left")

    # re-order columns
    ratings = ratings[['user_id', 'age', 'gender', 'occupation', 'movie_id', 'title', 'genres', 'year', 'rating', 'timestamp']]
    # save as a csv
    ratings.to_csv(f'{cf.PREPROCESSED_DATA_PATH}/ratings.csv', index=False)
    logger.info("Raw rating pre-processing finished.")

if __name__ == "__main__":
    movies = preprocess_movies()
    users = preprocess_users()
    preprocess_ratings(users, movies)