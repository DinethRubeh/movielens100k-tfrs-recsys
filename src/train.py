import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple
from models.user_model import UserModel
from models.movie_model import MovieModel
from models.ranking_model import RankingModel, RecommendationModel
from configuration import config as cf
from utils.log_utils import logger
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data() -> Tuple:
    # ratings data
    ratings_df = pd.read_csv(cf.RATINGS_DATA_PATH)

    # cast data type
    ratings_df["user_id"] = ratings_df["user_id"].astype(str)  # for StringLookup
    ratings_df["title"] = ratings_df["title"].astype(str) # for StringLookup
    ratings_df["rating"] = ratings_df["rating"].astype(float)
    logger.info("successfully loaded ratings dataframe")

    # convert to tf datasets
    ratings_tf_dataset = tf.data.Dataset.from_tensor_slices(dict(ratings_df))

    ratings_tf_dataset = ratings_tf_dataset.map(lambda x: {
        "movie_title": x["title"],
        "user_id": x["user_id"],
        "user_rating": x["rating"],
        "gender": x["gender"],
        "occupation": x["occupation"],
        "genres": x["genres"]
    })
    logger.info("ratings_df converted to tf dataset object")

    return ratings_df, ratings_tf_dataset

def train_test_split_ratings_data(ratings_tf_dataset):
    tf.random.set_seed(42)
    shuffled = ratings_tf_dataset.shuffle(cf.FULL_DATASET_SIZE, seed=42, reshuffle_each_iteration=False)

    # train/test split
    train = shuffled.take(cf.TRAIN_SIZE)
    test = shuffled.skip(cf.TRAIN_SIZE).take(cf.TEST_SIZE)
    logger.info("successfully train/test split the tf dataset")

    return train, test

def get_features_vocabs(ratings_df: pd.DataFrame, ratings_tf_dataset) -> Tuple:
    movie_titles = ratings_tf_dataset.batch(1_000_000).map(lambda x: x["movie_title"])
    user_ids = ratings_tf_dataset.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    unique_genders = ratings_df["gender"].dropna().unique().tolist()
    unique_occupations = ratings_df["occupation"].dropna().unique().tolist()
    unique_genres = ratings_df["genres"].dropna().unique().tolist()
    logger.info("successfully created the feature vocabularies")

    return (
        unique_movie_titles, unique_user_ids, 
        unique_genders, unique_occupations, unique_genres
    )

def train_model(train, feature_vocab):

    unique_movie_titles, unique_user_ids, unique_genders, unique_occupations, unique_genres = feature_vocab
    
    # load the defined ranking models
    user_model = UserModel(unique_user_ids, unique_genders, unique_occupations)
    movie_model = MovieModel(unique_movie_titles, unique_genres)
    ranking_model = RankingModel(user_model, movie_model)
    model = RecommendationModel(ranking_model)
    
    # compile
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=cf.LR))
    # shuffle, batch, and cache the training set
    cached_train = train.shuffle(cf.FULL_DATASET_SIZE).batch(cf.TRAIN_BATCH).cache()
    # fit the model
    model.fit(cached_train, epochs=cf.EPOCHS)
    logger.info("Model training process successful")

    return model

def evaluate(model, test) -> None:
    cached_test = test.batch(cf.TEST_BATCH).cache()
    logger.info("Evaluation results:")
    logger.info(model.evaluate(cached_test, return_dict=True))

def store_model(model) -> None:
    @tf.function(input_signature=[
        tf.TensorSpec([None], tf.string, name="user_id"),
        tf.TensorSpec([None], tf.string, name="gender"),
        tf.TensorSpec([None], tf.string, name="occupation"),
        tf.TensorSpec([None], tf.string, name="movie_title"),
        tf.TensorSpec([None], tf.string, name="genres"),
    ])
    def serve_fn(user_id, gender, occupation, movie_title, genres):
        features = {
            "user_id": user_id,
            "gender": gender,
            "occupation": occupation,
            "movie_title": movie_title,
            "genres": genres,
        }
        return model(features)

    tf.saved_model.save(model, export_dir=cf.MODEL_PATH, signatures={"serving_default": serve_fn})
    
    logger.info("Model successfully saved.")

def main():
    ratings_df, ratings_tf_dataset = load_data()

    train_set, test_set = train_test_split_ratings_data(ratings_tf_dataset)

    feature_vocab = get_features_vocabs(ratings_df, ratings_tf_dataset)

    model = train_model(train_set, feature_vocab)

    evaluate(model, test_set)

    store_model(model)

if __name__ == "__main__":
    main()