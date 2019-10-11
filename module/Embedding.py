import tensorflow as tf
import tensorflow.keras as keras

class Embedding(keras.layers.Layer):

    def __init__(self, input_size,
                 output_size,
                 weights=None):
        super(Embedding, self).__init__()

        if weights is not None:
            self.embedding = tf.keras.layers.Embedding(input_size, output_size, weights=weights, mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(input_size, output_size, mask_zero=True)

    def __call__(self, input):  # [batch, len]

        return self.embedding(input)  # [batch, len, output_size]


