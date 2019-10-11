import tensorflow.keras as keras

class Decoder(keras.layers.Layer):

    def __init__(self, rnn_type,
                 input_size,
                 output_size,
                 num_layers):
        super(Decoder, self).__init__()
        assert rnn_type in ['GRU', 'LSTM']

        if rnn_type == 'GRU':
            rnnCell = [getattr(keras.layers, 'GRUCell')(output_size) for _ in range(num_layers)]
        else:
            rnnCell = [getattr(keras.layers, 'LSTMCell')(output_size) for _ in range(num_layers)]

        self.rnn = keras.layers.RNN(rnnCell, input_shape=(None, None, input_size),
                                    return_sequences=True, return_state=True)
        self.rnn_type = rnn_type
        self.num_layers = num_layers

    def __call__(self, input, states):  # input: [batch, 1, input_size]

        outputs = self.rnn(input, states)

        # output: [batch, 1, output_size]
        return outputs[0], outputs[1:]
