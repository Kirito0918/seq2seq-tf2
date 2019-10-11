import tensorflow as tf
import tensorflow.keras as keras

class Encoder(keras.layers.Layer):

    def __init__(self, rnn_type,  # rnn类型
                 input_size,
                 output_size,
                 num_layers,  # rnn层数
                 bidirectional=False):
        super(Encoder, self).__init__()
        assert rnn_type in ['GRU', 'LSTM']
        if bidirectional:
            assert output_size % 2 == 0

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        units = int(output_size / self.num_directions)

        if rnn_type == 'GRU':
            rnnCell = [getattr(keras.layers, 'GRUCell')(units) for _ in range(num_layers)]
        else:
            rnnCell = [getattr(keras.layers, 'LSTMCell')(units) for _ in range(num_layers)]

        self.rnn = keras.layers.RNN(rnnCell, input_shape=(None, None, input_size),
                                    return_sequences=True, return_state=True)
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        if bidirectional:
            self.rnn = keras.layers.Bidirectional(self.rnn, merge_mode='concat')

        self.bidirectional = bidirectional


    def __call__(self, input):  # [batch, timesteps, input_dim]

        outputs = self.rnn(input)

        output = outputs[0]
        states = outputs[1:]
        states_forward = states[: self.num_layers]
        states_backward = states[self.num_layers:]

        if self.bidirectional:

            states = []

            if self.rnn_type == 'LSTM':
                for idx in range(self.num_layers):
                    state_hf, state_cf = states_forward[idx]
                    state_hb, state_cb = states_backward[idx]
                    state_h = tf.concat([state_hf, state_hb], 1)
                    state_c = tf.concat([state_cf, state_cb], 1)
                    state = [state_h, state_c]
                    states.append(state)

            else:  # 'GRU'
                for idx in range(self.num_layers):
                    state_hf = states_forward[idx]
                    state_hb = states_backward[idx]
                    state = tf.concat([state_hf, state_hb], 1)
                    states.append(state)

        # output: [batch_size, encoder_len, output_size]
        # states: [num_layers] * tensor(batch, output_size)
        return output, states
