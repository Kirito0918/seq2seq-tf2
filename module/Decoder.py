import tensorflow.keras as keras

class Decoder(keras.layers.Layer):

    def __init__(self, rnn_type,
                 output_dim,
                 num_layers):
        super(Decoder, self).__init__()
        assert rnn_type in ['GRU', 'LSTM']

        self.rnns = [getattr(keras.layers, rnn_type)(output_dim, return_sequences=True, return_state=True)] * num_layers
        self.rnn_type = rnn_type
        self.num_layers = num_layers

    def __call__(self, input, state):  # [batch, timesteps, input_dim]

        for rnn in self.rnns:

            output, state = rnn(input, state)

            input = output

        # output: [batch, seq, dim*dircetions]
        return output