import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text

# ranking model
class RankingModel(tf.keras.Model):
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        user_repr = self.user_model({
            "user_id": inputs["user_id"],
            "gender": inputs["gender"],
            "occupation": inputs["occupation"]
        })

        movie_repr = self.movie_model({
            "movie_title": inputs["movie_title"],
            "genres": inputs["genres"]
        })

        return self.mlp(tf.concat([user_repr, movie_repr], axis=1))
  
# full model
class RecommendationModel(tfrs.models.Model):
    def __init__(self, ranking_model):
        super().__init__()
        self.ranking_model = ranking_model
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features):
        return self.ranking_model(features)

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features["user_rating"]
        inputs = {key: features[key] for key in features if key != "user_rating"}

        rating_predictions = self(inputs)

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)