import tensorflow as tf

class MovieModel(tf.keras.Model):
    def __init__(self, unique_movie_titles, unique_genres):
        super().__init__()

        max_tokens = 10_000

        # Movie title embedding
        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32)
        ])

        # Movie title text embedding
        self.title_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
        self.title_text_embedding = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

        # Genre text vectorization and embedding
        self.genre_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
        self.genre_embedding = tf.keras.Sequential([
            self.genre_vectorizer,
            tf.keras.layers.Embedding(max_tokens, 16, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D()
        ])

        # Adapt vectorizers
        self.genre_vectorizer.adapt(unique_genres)
        self.title_vectorizer.adapt(unique_movie_titles)

    def call(self, inputs):  # inputs is a dict with 'movie_title' and 'genres'
        return tf.concat([
            self.title_embedding(inputs["movie_title"]),
            self.title_text_embedding(inputs["movie_title"]),
            self.genre_embedding(inputs["genres"])
        ], axis=1)