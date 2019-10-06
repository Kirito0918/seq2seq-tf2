import tensorflow as tf

class Embedding(tf.keras.layers.Layer):

    def __init__(self, input_dim,
                 output_dim,
                 pad_id):
        self.embedding = tf.keras.layers.Embedding()
