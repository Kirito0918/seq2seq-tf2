import tensorflow as tf
import tensorflow.keras as keras

class Embedding(keras.layers.Layer):

    def __init__(self, input_dim,
                 output_dim,
                 weights):
        super(Embedding, self).__init__()

        if weights is not None:
            self.embedding = tf.keras.layers.Embedding(input_dim, output_dim, weights=weights, mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(input_dim, output_dim, mask_zero=True)

    def __call__(self, input):  # [batch, vocab]

        return self.embedding(input)


