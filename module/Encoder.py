import tensorflow as tf
import tensorflow.keras as keras

class Encoder(keras.layers.Layer):

    def __init__(self, rnn_type,
                 output_dim,
                 num_layers,
                 bidirectional=False):
        super(Encoder, self).__init__()
        assert rnn_type in ['GRU', 'LSTM']
        if bidirectional:
            assert output_dim % 2 == 0

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        units = int(output_dim / self.num_directions)

        self.rnns = [getattr(keras.layers, rnn_type)(units, return_sequences=True, return_state=True)] * num_layers
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        if bidirectional:
            self.rnns = [keras.layers.Bidirectional(rnn, merge_mode='concat') for rnn in self.rnns]

        self.bidirectional = bidirectional


    def __call__(self, input):  # [batch, timesteps, input_dim]

        states = []

        for idx, rnn in enumerate(self.rnns):

            if idx == 0:
                state = None

            if self.rnn_type == 'LSTM':
                if self.bidirectional:
                    output, state_hf, state_cf, state_hb, state_cb = rnn(input, state)
                    state = (state_hf, state_cf, state_hb, state_cb)
                else:
                    output, state_h, state_c = rnn(input, state)
                    state = (state_h, state_c)
            else:  # gru
                if self.bidirectional:
                    output, state_hf, state_hb = rnn(input, state)
                    state = (state_hf, state_hb)
                else:
                    output, state_h = rnn(input, state)
                    state = state_h

            states.append(state)
            input = output

        # output: [batch, seq, dim*dircetions]
        # states: [num_layers] * (state) * tensor(batch, dim)
        return output, states







