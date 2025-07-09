import tensorflow as tf

class UserModel(tf.keras.Model):
    def __init__(self, unique_user_ids, unique_genders, unique_occupations):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32)
        ])

        self.gender_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_genders, mask_token=None),
            tf.keras.layers.Embedding(len(unique_genders) + 1, 8)
        ])

        self.occupation_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_occupations, mask_token=None),
            tf.keras.layers.Embedding(len(unique_occupations) + 1, 8)
        ])

    def call(self, inputs):  # inputs is a dict with 'user_id', 'gender', 'occupation'
        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.gender_embedding(inputs["gender"]),
            self.occupation_embedding(inputs["occupation"])
        ], axis=1)