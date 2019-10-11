import tensorflow.keras as keras

class Decoder(keras.layers.Layer):

    def __init__(self, rnn_type,
                 input_size,
                 output_size,
                 num_layers):
        super(Decoder, self).__init__()
        assert rnn_type in ['GRU', 'LSTM']

        self.rnns = [getattr(keras.layers, rnn_type)(output_size, input_shape=(None, None, input_size),
                                                     return_sequences=True, return_state=True)
                     for _ in range(num_layers)]
        self.rnn_type = rnn_type
        self.num_layers = num_layers

    def __call__(self, input, states):  # input: [batch, 1, input_size]

        new_states = []

        for idx, rnn in enumerate(self.rnns):

            outputs = rnn(input, states[idx])

            input = outputs[0]

            new_states.append(tuple(outputs[1:]))

        # output: [batch, 1, output_size]
        return outputs[0], new_states
